"""Phase 5A.2 isolated web-search probe: ONE DuckDuckGo query, macOS system DNS.

No LLM, no MCP, no LangGraph. It runs ONLY the exact plan in plan_phase5a2.json and only
with the approval code derived from that plan's SHA-256 digest (any change to the query or
configuration changes the code).

Provider path (verified in Phase 5A.1, no package edits):
  ddgs 9.16.1 Duckduckgo engine  ->  injected primp 2.0.1 Client(dns_resolver="system")
The ddgs multi-provider orchestrator (DDGS / backend="auto") is never used, so there is no
provider fallback. One HTTP request per query, no application-level retries, stop on failure.

  dry run (no network):  .venv/bin/python -I search_lab/ddgs_probe.py
  approved run:          ./search_lab/run_search_sandboxed.sh --approved <CODE>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import urllib.request
from datetime import datetime
from importlib import metadata
from pathlib import Path
from urllib.parse import urlsplit

PROXY_ENV = ("PRIMP_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY",
             "http_proxy", "https_proxy", "all_proxy", "ftp_proxy")
for _name in (*PROXY_ENV, "OLLAMA_API_KEY", "OLLAMA_HOST", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
    os.environ.pop(_name, None)

HERE = Path(__file__).resolve().parent
PLAN_FILE = HERE / "plan_phase5a2.json"
RESULTS_DIR = HERE.parent / "results"

REQUIRED_VERSIONS = {"ddgs": "9.16.1", "primp": "2.0.1"}
HOSTNAME = "html.duckduckgo.com"
SEARCH_URL = f"https://{HOSTNAME}/html/"
BACKEND = "duckduckgo"
DNS_MODE = "system"
MAX_QUERIES = 1
MAX_QUERY_CHARS = 200
MAX_RESULTS = 5
TIMEOUT_S = 10
MAX_TITLE_CHARS, MAX_SNIPPET_CHARS, MAX_URL_CHARS = 300, 500, 2_048

# The exact plan fields and values a human approves (the query is the only free field).
PLAN_CONSTANTS = {
    "provider": "DuckDuckGo", "hostname": HOSTNAME, "port": 443, "dns_mode": "macOS system resolver",
    "backend": BACKEND, "region": "us-en", "max_results": MAX_RESULTS, "timeout_s": TIMEOUT_S,
    "app_retries": 0, "other_providers": "disabled", "paid_apis": "none",
}

_ABS_PATH = re.compile(r"(?<![\w.:/])(?:~?/[^\s\"'<>]+|[A-Za-z]:\\[^\s\"'<>]+)")  # leaves URLs intact
_FORBIDDEN_QUERY = [
    ("local path", re.compile(r"(?:^|\s)(?:~|/|[A-Za-z]:\\)|\.\./")),
    ("email address", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("URL", re.compile(r"https?://|www\.", re.I)),
    ("secret-like token", re.compile(r"[A-Za-z0-9_\-+/=]{24,}")),
    ("credential word", re.compile(r"\b(?:password|passwd|api[_ -]?key|secret|token|bearer)\b", re.I)),
]
_PROXY_FLAGS = ("HTTPEnable", "HTTPSEnable", "SOCKSEnable", "FTPEnable", "GopherEnable", "RTSPEnable",
                "ProxyAutoConfigEnable", "ProxyAutoDiscoveryEnable")


class IncompatibleLibrary(RuntimeError):
    """Installed ddgs/primp differ from what Phase 5A.1 verified; refuse to run."""


class RatelimitException(Exception):
    pass


class DDGSException(Exception):
    pass


# --- plan and approval ------------------------------------------------------------------
def query_problem(query: object) -> str | None:
    if not isinstance(query, str) or not query.strip():
        return "empty or not text"
    if len(query) > MAX_QUERY_CHARS:
        return "too long"
    if any(unicodedata.category(c).startswith("C") for c in query):
        return "control or invisible characters"
    for label, rx in _FORBIDDEN_QUERY:
        if rx.search(query):
            return f"contains a {label}"
    return None


def load_plan(path: Path = PLAN_FILE) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if set(plan) != {*PLAN_CONSTANTS, "queries"}:
        raise ValueError("plan fields differ from the approved schema")
    for key, value in PLAN_CONSTANTS.items():
        if plan[key] != value:
            raise ValueError(f"plan field {key!r} must be {value!r}")
    queries = plan["queries"]
    if not isinstance(queries, list) or len(queries) != MAX_QUERIES:
        raise ValueError(f"plan must contain exactly {MAX_QUERIES} query")
    for q in queries:
        problem = query_problem(q)
        if problem:
            raise ValueError(f"query rejected ({problem}): {q!r}")
    return plan


def plan_digest(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def approval_code(plan: dict) -> str:
    return plan_digest(plan)[:8].upper()


# --- preflight (no network) ----------------------------------------------------------------
def _installed_version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def check_versions(version_of=None) -> None:
    version_of = version_of or _installed_version  # looked up at call time
    found = {name: version_of(name) for name in REQUIRED_VERSIONS}
    if found != REQUIRED_VERSIONS:
        raise IncompatibleLibrary(f"need {REQUIRED_VERSIONS}, found {found}")


def proxy_problems(getproxies=urllib.request.getproxies, scutil=None) -> list[str]:
    """Any proxy would change where traffic goes; refuse if one is configured."""
    problems = [f"environment variable {n} is set" for n in PROXY_ENV if os.environ.get(n)]
    active = {k: v for k, v in getproxies().items() if k != "no" and v}
    if active:
        problems.append(f"system/environment proxies active: {sorted(active)}")
    try:
        text = scutil() if scutil else subprocess.run(["scutil", "--proxy"], capture_output=True, text=True,
                                                      timeout=5, check=True).stdout
    except Exception as exc:  # noqa: BLE001 - fail closed
        return [*problems, f"cannot read macOS proxy settings ({type(exc).__name__})"]
    for flag in _PROXY_FLAGS:
        if re.search(rf"^\s*{flag}\s*:\s*1\s*$", text, re.M):
            problems.append(f"macOS proxy setting {flag} is on")
    return problems


def preflight(version_of=None, getproxies=urllib.request.getproxies, scutil=None) -> list[str]:
    try:
        check_versions(version_of)
    except IncompatibleLibrary as exc:
        return [str(exc)]
    return proxy_problems(getproxies, scutil)


# --- provider client: pinned engine + system DNS (no network during construction) ----------
def build_engine():
    """Return the ddgs Duckduckgo engine with its HTTP client replaced by a system-DNS client."""
    check_versions()
    import ddgs
    import primp
    from ddgs.engines.duckduckgo import Duckduckgo
    from ddgs.http_client import HttpClient

    if ddgs.__version__ != REQUIRED_VERSIONS["ddgs"]:
        raise IncompatibleLibrary("ddgs module version mismatch")
    expected = {"name": BACKEND, "category": "text", "search_url": SEARCH_URL, "search_method": "POST"}
    for attr, value in expected.items():
        if getattr(Duckduckgo, attr, None) != value:
            raise IncompatibleLibrary(f"Duckduckgo.{attr} is not {value!r}")
    engine = Duckduckgo(proxy=None, timeout=TIMEOUT_S, verify=True)
    http = getattr(engine, "http_client", None)
    if not isinstance(http, HttpClient) or not isinstance(getattr(http, "client", None), primp.Client):
        raise IncompatibleLibrary("unexpected ddgs engine/http_client structure")
    if not callable(getattr(http, "request", None)) or not isinstance(getattr(engine, "headers_update", None), dict):
        raise IncompatibleLibrary("unexpected ddgs engine/http_client structure")
    client = primp.Client(proxy=None, timeout=TIMEOUT_S, impersonate="random", impersonate_os="random",
                          verify=True, dns_resolver=DNS_MODE)  # same as ddgs, except the resolver
    client.headers_update(engine.headers_update)
    http.client = client
    if engine.http_client.client is not client:
        raise IncompatibleLibrary("client injection did not take effect")
    return engine


class PinnedDuckDuckGo:
    """`.text()` over one engine; records every HTTP request the engine makes."""

    def __init__(self, engine):
        self.engine, self.requests = engine, []
        original = engine.http_client.request

        def recording_request(method, url, *args, **kwargs):
            entry = {"method": method, "url": url}
            self.requests.append(entry)
            resp = original(method, url, *args, **kwargs)
            entry["status"] = getattr(resp, "status_code", None)
            entry["final_url"] = str(getattr(getattr(resp, "_resp", None), "url", "") or "")
            entry["bytes"] = len(getattr(resp, "content", b"") or b"")
            return resp

        engine.http_client.request = recording_request

    def text(self, query: str, *, region: str, max_results: int, backend: str) -> list[dict]:
        if backend != BACKEND:
            raise ValueError("only the pinned duckduckgo backend is allowed")
        results = self.engine.search(query, region=region, safesearch="moderate", timelimit=None, page=1)
        if results is None:  # ddgs returns None for any non-200 response
            status = self.requests[-1].get("status") if self.requests else None
            if status in (202, 429):
                raise RatelimitException(f"HTTP {status}")
            raise DDGSException(f"HTTP {status}")
        return [{"title": getattr(r, "title", ""), "href": getattr(r, "href", ""), "body": getattr(r, "body", "")}
                for r in results][:max_results]


def make_client() -> PinnedDuckDuckGo:
    return PinnedDuckDuckGo(build_engine())


# --- result handling (results are untrusted third-party text) ------------------------------
def clean_text(value: object, limit: int) -> str:
    text = "".join(" " if unicodedata.category(c).startswith("C") else c for c in str(value))
    return " ".join(text.split())[:limit]


def safe_url(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > MAX_URL_CHARS or any(c.isspace() for c in value):
        return None
    if any(unicodedata.category(c).startswith("C") for c in value):
        return None
    parts = urlsplit(value)
    return value if parts.scheme in ("http", "https") and parts.netloc else None


def normalize(raw: object) -> tuple[list[dict], int]:
    hits, dropped = [], 0
    for item in raw if isinstance(raw, list) else []:
        url = safe_url(item.get("href")) if isinstance(item, dict) else None
        if url is None:
            dropped += 1
            continue
        hits.append({"title": clean_text(item.get("title", ""), MAX_TITLE_CHARS), "url": url,
                     "snippet": clean_text(item.get("body", ""), MAX_SNIPPET_CHARS)})
    return hits[:MAX_RESULTS], dropped


def classify_error(exc: BaseException) -> str:
    name = type(exc).__name__
    if name == "TimeoutException" or isinstance(exc, TimeoutError):
        return "timeout"
    if name == "RatelimitException":
        return "rate_limited"
    if name == "DDGSException":
        return "empty_or_blocked" if "no results" in str(exc).lower() else "provider_error"
    if isinstance(exc, IncompatibleLibrary):
        return "incompatible_library"
    return "unexpected_error"


def run_plan(plan: dict, approved_code: str, client_factory=make_client) -> list[dict]:
    if approved_code != approval_code(plan):
        raise PermissionError("approval code does not match this exact plan; nothing was sent")
    records = []
    try:
        client = client_factory()
    except Exception as exc:  # noqa: BLE001
        return [{"query": plan["queries"][0], "status": classify_error(exc), "error_type": type(exc).__name__,
                 "error": _ABS_PATH.sub("<path>", str(exc))[:300], "results": [], "requests": [], "seconds": 0.0,
                 "fields": {"title": 0, "url": 0, "snippet": 0}}]
    for query in plan["queries"][:MAX_QUERIES]:
        start = time.perf_counter()
        record = {"query": query, "provider": "DuckDuckGo", "backend": BACKEND, "dns_mode": DNS_MODE}
        try:
            raw = client.text(query, region=plan["region"], max_results=plan["max_results"], backend=BACKEND)
            hits, dropped = normalize(raw)
            record.update(status="ok" if hits else "empty", results=hits, dropped=dropped)
        except Exception as exc:  # noqa: BLE001 - classify, never retry
            record.update(status=classify_error(exc), error_type=type(exc).__name__,
                          error=_ABS_PATH.sub("<path>", str(exc))[:300], results=[])
        record["seconds"] = round(time.perf_counter() - start, 2)
        record["requests"] = list(getattr(client, "requests", []))
        record["fields"] = {k: sum(bool(h[k]) for h in record["results"]) for k in ("title", "url", "snippet")}
        records.append(record)
        if record["status"] != "ok":
            break
    return records


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--approved", help="approval code shown by the dry run")
    args = p.parse_args(argv)
    plan = load_plan()
    print(json.dumps(plan, indent=2))
    print(f"plan sha256: {plan_digest(plan)}\nAPPROVAL CODE: {approval_code(plan)}")
    problems = preflight()
    print("preflight:", "OK (versions exact, no proxies)" if not problems else problems)
    if problems:
        return 2
    if not args.approved:
        print("dry run only: nothing was sent")
        return 0
    records = run_plan(plan, args.approved.strip().upper())
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"search_probe_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps({**r, "plan_sha256": plan_digest(plan)}) + "\n")
    for r in records:
        print(f"\n[{r['status']}] {r['query']!r}  {r['seconds']}s  results={len(r['results'])} "
              f"dropped={r.get('dropped', 0)} fields={r['fields']} {r.get('error_type', '')} {r.get('error', '')}")
        print("http requests:", json.dumps(r.get("requests", [])))
        for h in r["results"]:
            print(f"  - {h['title'][:100]!r}\n    {h['url'][:120]}\n    {h['snippet'][:200]!r}")
    print(f"\nqueries sent: {len(records)}; log: {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
