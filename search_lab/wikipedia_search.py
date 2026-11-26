"""Phase 5A.3 isolated adapter for Wikipedia's official MediaWiki Action API (list=search).

Standard library only. No LLM, MCP or LangGraph. It sends ONE GET request for the exact plan
in plan_phase5a3.json, and only with the approval code derived from that plan's SHA-256
digest (query, User-Agent and every setting are bound to the code).

Safety properties (application-enforced; the OS sandbox adds HTTPS-only egress):
  * HTTPS only, pinned host en.wikipedia.org and path /w/api.php, fixed parameters
  * system DNS (urllib -> getaddrinfo -> macOS resolver); proxies disabled
  * TLS certificate and hostname verification; redirects refused (any 3xx is an error)
  * 10 s socket timeout, response capped before parsing, no retries of any kind
  * policy-compliant User-Agent with operator-supplied contact (never invented here)
  * results are untrusted: HTML stripped, text cleaned and bounded, URLs built locally

  dry run (no network):  .venv/bin/python -I search_lab/wikipedia_search.py
  approved run:          ./search_lab/run_wiki_sandboxed.sh --approved <CODE>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import ssl
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit

HERE = Path(__file__).resolve().parent
PLAN_FILE = HERE / "plan_phase5a3.json"
RESULTS_DIR = HERE.parent / "results"

HOST = "en.wikipedia.org"
PATH = "/w/api.php"
ENDPOINT = f"https://{HOST}{PATH}"
ARTICLE_BASE = f"https://{HOST}/wiki/"
FIXED_PARAMS = {  # documented on mediawiki.org API:Search; srprop limited to the snippet only
    "action": "query", "list": "search", "format": "json", "formatversion": "2",
    "srlimit": "5", "srnamespace": "0", "srprop": "snippet", "utf8": "1",
}
MAX_QUERY_CHARS = 200
MAX_RESULTS = 5
TIMEOUT_S = 10
MAX_RESPONSE_BYTES = 65_536
MAX_TITLE_CHARS, MAX_SNIPPET_CHARS = 300, 500

PLAN_CONSTANTS = {
    "provider": "Wikipedia (MediaWiki Action API)", "hostname": HOST, "port": 443, "path": PATH,
    "method": "GET", "dns_mode": "macOS system resolver", "params": FIXED_PARAMS,
    "max_results": MAX_RESULTS, "timeout_s": TIMEOUT_S, "max_response_bytes": MAX_RESPONSE_BYTES,
    "redirects": "refused", "app_retries": 0, "other_providers": "disabled", "paid_apis": "none",
}

_FORBIDDEN_QUERY = [
    ("local path", re.compile(r"(?:^|\s)(?:~|/|[A-Za-z]:\\)|\.\./")),
    ("email address", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("URL", re.compile(r"https?://|www\.", re.I)),
    ("secret-like token", re.compile(r"[A-Za-z0-9_\-+/=]{24,}")),
    ("credential word", re.compile(r"\b(?:password|passwd|api[_ -]?key|secret|token|bearer)\b", re.I)),
    ("search-syntax keyword", re.compile(r"\b[a-z]+:", re.I)),        # CirrusSearch: insource:, intitle:, ...
    ("regex or special search syntax", re.compile(r"[/\\~*?{}\[\]|]")),
]
# <client>/<version> (<contact>) <library>/<version>   -- Wikimedia User-Agent policy
_UA = re.compile(r"(?P<client>[A-Za-z][A-Za-z0-9._-]{2,40})/(?P<ver>[0-9][0-9A-Za-z.]{0,15}) "
                 r"\((?P<contact>[^()\r\n]{5,200})\)(?: [A-Za-z][A-Za-z0-9._-]{1,40}/[0-9][0-9A-Za-z.]{0,15})*")
_CONTACT = re.compile(r"https://[^\s;()]+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\w+:\w+; User:[^;()]+")
_PLACEHOLDER = re.compile(r"example\.(?:org|com|net)|\.invalid\b|\.test\b|\.localhost\b|localhost|"
                          r"your[-_ ]?(?:email|name|site)|changeme|todo|xxx|@example", re.I)


# --- validation -----------------------------------------------------------------------------
def query_problem(query: object) -> str | None:
    if not isinstance(query, str) or not query.strip():
        return "empty or not text"
    if len(query) > MAX_QUERY_CHARS:
        return "too long"
    if any(unicodedata.category(c)[0] in "CZ" and c != " " for c in query):
        return "control, invisible or unusual whitespace characters"
    for label, rx in _FORBIDDEN_QUERY:
        if rx.search(query):
            return f"contains a {label}"
    return None


def user_agent_problem(ua: object, *, require_real_contact: bool = True) -> str | None:
    if not isinstance(ua, str) or not _UA.fullmatch(ua):
        return "User-Agent must be '<client>/<version> (<contact>) <library>/<version>'"
    m = _UA.fullmatch(ua)
    if m["client"].lower() in ("python-urllib", "python-requests", "curl", "wget", "mozilla", "lwp"):
        return "generic or browser client names are not allowed"
    if not _CONTACT.search(m["contact"]):
        return "contact must be an https URL, an email address, or '<project>; User:<name>'"
    if require_real_contact and _PLACEHOLDER.search(m["contact"]):
        return "contact looks like a placeholder; use real operator contact information"
    return None


def load_plan(path: Path | None = None) -> dict:
    plan = json.loads((path or PLAN_FILE).read_text(encoding="utf-8"))  # looked up at call time
    if set(plan) != {*PLAN_CONSTANTS, "query", "user_agent"}:
        raise ValueError("plan fields differ from the approved schema")
    for key, value in PLAN_CONSTANTS.items():
        if plan[key] != value:
            raise ValueError(f"plan field {key!r} must be {value!r}")
    problem = query_problem(plan["query"]) or user_agent_problem(plan["user_agent"])
    if problem:
        raise ValueError(problem)
    return plan


def plan_digest(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def approval_code(plan: dict) -> str:
    return plan_digest(plan)[:8].upper()


def preflight() -> list[str]:
    """Local checks only (no network)."""
    problems = []
    try:
        ctx = ssl.create_default_context()
        if ctx.verify_mode != ssl.CERT_REQUIRED or not ctx.check_hostname:
            problems.append("TLS verification is not enforced")
        if not ctx.cert_store_stats().get("x509_ca"):
            ctx.load_default_certs()
            if not ctx.cert_store_stats().get("x509_ca"):
                problems.append("no trusted CA certificates available to Python")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"TLS setup failed ({type(exc).__name__})")
    return problems


# --- HTTP (one request, no proxies, no redirects) -------------------------------------------
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401, PLR0913
        return None  # urllib then raises HTTPError for the 3xx


def build_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()  # CERT_REQUIRED + hostname check
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect(),
                                       urllib.request.HTTPSHandler(context=ctx))


def build_request(plan: dict) -> urllib.request.Request:
    url = f"{ENDPOINT}?{urlencode({**FIXED_PARAMS, 'srsearch': plan['query']})}"
    parts = urlsplit(url)
    if (parts.scheme, parts.hostname, parts.port, parts.path) != ("https", HOST, None, PATH):
        raise ValueError("request URL is not the pinned endpoint")
    return urllib.request.Request(url, method="GET", headers={
        "User-Agent": plan["user_agent"], "Accept": "application/json", "Accept-Encoding": "identity"})


# --- result handling (untrusted third-party text) --------------------------------------------
class _TextOnly(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(fragment: object, limit: int) -> str:
    parser = _TextOnly()
    parser.feed(str(fragment))
    parser.close()
    return clean_text("".join(parser.parts), limit)


def clean_text(value: object, limit: int) -> str:
    text = "".join(" " if unicodedata.category(c)[0] in "CZ" else c for c in str(value))
    return " ".join(text.split())[:limit]


def article_url(title: str) -> str:
    return ARTICLE_BASE + quote(title.replace(" ", "_"), safe="()_,:'!-.~")


def parse_results(payload: object) -> tuple[list[dict], int]:
    if not isinstance(payload, dict):
        raise _Fail("unexpected_shape", "top-level JSON is not an object")
    if "error" in payload:
        err = payload["error"] if isinstance(payload["error"], dict) else {}
        code = clean_text(err.get("code", "unknown"), 60)
        raise _Fail("rate_limited" if code == "ratelimited" else "api_error", f"API error code {code}")
    items = payload.get("query", {}).get("search") if isinstance(payload.get("query"), dict) else None
    if not isinstance(items, list):
        raise _Fail("unexpected_shape", "query.search is missing or not a list")
    hits, dropped = [], 0
    for item in items:
        ok = (isinstance(item, dict) and item.get("ns") == 0 and isinstance(item.get("title"), str)
              and isinstance(item.get("pageid"), int) and item["pageid"] > 0)
        title = clean_text(item["title"], MAX_TITLE_CHARS) if ok else ""
        if not title:
            dropped += 1
            continue
        hits.append({"title": title, "url": article_url(title), "pageid": item["pageid"],
                     "snippet": html_to_text(item.get("snippet", ""), MAX_SNIPPET_CHARS)})
    return hits[:MAX_RESULTS], dropped


class _Fail(Exception):
    def __init__(self, status: str, detail: str, **extra):
        super().__init__(detail)
        self.status, self.detail, self.extra = status, detail, extra


def _http_status(code: int) -> str:
    if 300 <= code < 400:
        return "redirect_refused"
    return {403: "forbidden", 429: "rate_limited", 503: "unavailable"}.get(
        code, "server_error" if code >= 500 else "http_error")


def search_once(plan: dict, opener=None) -> dict:
    """Exactly one request. Never retries. Returns a record; never raises for network errors."""
    started, record = time.perf_counter(), {"query": plan["query"], "provider": PLAN_CONSTANTS["provider"],
                                            "host": HOST, "method": "GET", "requests": 0}
    try:
        request = build_request(plan)
        record["requests"] = 1
        with (opener or build_opener()).open(request, timeout=TIMEOUT_S) as resp:
            final = urlsplit(resp.geturl())
            record["http_status"] = resp.status
            if (final.scheme, final.hostname) != ("https", HOST):
                raise _Fail("redirect_refused", "response came from another location")
            ctype = resp.headers.get("Content-Type", "")
            if not ctype.lower().startswith("application/json"):
                raise _Fail("bad_content_type", f"unexpected Content-Type {clean_text(ctype, 60)!r}")
            body = resp.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise _Fail("response_too_large", f"more than {MAX_RESPONSE_BYTES} bytes")
        record["bytes"] = len(body)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise _Fail("malformed_json", "response is not valid UTF-8 JSON") from None
        if isinstance(payload, dict) and payload.get("warnings"):
            record["api_warnings"] = clean_text(json.dumps(payload["warnings"]), 300)
        hits, dropped = parse_results(payload)
        record.update(status="ok" if hits else "empty", results=hits, dropped=dropped)
    except _Fail as f:
        record.update(status=f.status, error=f.detail, results=[], **f.extra)
    except urllib.error.HTTPError as e:
        record.update(status=_http_status(e.code), http_status=e.code, results=[],
                      error=f"HTTP {e.code}", retry_after=clean_text(e.headers.get("Retry-After", ""), 20) or None)
    except urllib.error.URLError as e:
        reason = e.reason
        status = ("timeout" if isinstance(reason, (TimeoutError, socket.timeout)) else
                  "dns_error" if isinstance(reason, socket.gaierror) else
                  "tls_error" if isinstance(reason, ssl.SSLError) else "connection_error")
        record.update(status=status, error=clean_text(f"{type(reason).__name__}: {reason}", 200), results=[])
    except (TimeoutError, socket.timeout):
        record.update(status="timeout", error="timed out", results=[])
    except ssl.SSLError as e:
        record.update(status="tls_error", error=clean_text(type(e).__name__, 60), results=[])
    except Exception as e:  # noqa: BLE001 - fail closed, no retry
        record.update(status="unexpected_error", error=clean_text(type(e).__name__, 60), results=[])
    record["seconds"] = round(time.perf_counter() - started, 2)
    record["fields"] = {k: sum(bool(h[k]) for h in record["results"]) for k in ("title", "url", "snippet")}
    return record


def run_plan(plan: dict, approved_code: str, opener=None) -> dict:
    if approved_code != approval_code(plan):
        raise PermissionError("approval code does not match this exact plan; nothing was sent")
    problem = query_problem(plan["query"]) or user_agent_problem(plan["user_agent"], require_real_contact=False)
    if problem:
        raise ValueError(problem)
    return search_once(plan, opener)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--approved", help="approval code shown by the dry run")
    args = p.parse_args(argv)
    if not PLAN_FILE.exists():
        print(f"no plan file ({PLAN_FILE.name}); create it with an operator-approved User-Agent first")
        return 2
    plan = load_plan()
    print(json.dumps(plan, indent=2))
    print(f"plan sha256: {plan_digest(plan)}\nAPPROVAL CODE: {approval_code(plan)}")
    print(f"request URL: {build_request(plan).full_url}")
    problems = preflight()
    print("preflight:", "OK" if not problems else problems)
    if problems:
        return 2
    if not args.approved:
        print("dry run only: nothing was sent")
        return 0
    record = run_plan(plan, args.approved.strip().upper())
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"wiki_search_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({**record, "plan_sha256": plan_digest(plan)}, indent=2))
    print(f"\n[{record['status']}] {record['query']!r} {record['seconds']}s http={record.get('http_status')} "
          f"bytes={record.get('bytes')} results={len(record['results'])} dropped={record.get('dropped', 0)} "
          f"fields={record['fields']} {record.get('error', '')}")
    for h in record["results"]:
        print(f"  - {h['title']!r}\n    {h['url']}\n    {h['snippet'][:220]!r}")
    print(f"log: {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
