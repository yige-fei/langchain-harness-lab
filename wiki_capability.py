"""Phase 5B: the `wiki_search` capability, shared by the agent graph and the supervisor.

Standard library only (no LangChain, model client, MCP or network calls), so the unsandboxed
supervisor can import it without pulling in anything that talks to a network.

It defines exactly what a human approves for a Wikipedia search (the search spec + query), the
approval digest, how a search record from the worker is normalized, and how the final answer is
checked and given its source list.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
_SEARCH_LAB = str(LAB / "search_lab")
if _SEARCH_LAB not in sys.path:
    sys.path.append(_SEARCH_LAB)
import wikipedia_search as wiki  # noqa: E402  (the verified Phase 5A.3 adapter; functions/constants only)

SEARCH_TOOL = "wiki_search"
CONFIG_FILE = LAB / "search_lab" / "wiki_agent_config.json"
MAX_RESULTS = wiki.MAX_RESULTS          # 5
TIMEOUT_S = wiki.TIMEOUT_S              # 10
TRANSMITTED = ("The search query text (in the request URL), the User-Agent header shown above, "
               "Accept/Accept-Encoding headers, and your network's public IP address and TLS metadata. "
               "Nothing else: no files, no file contents, no conversation history, no email address, no credentials.")
LICENSE_NOTE = "text from Wikipedia, CC BY-SA 4.0"
DIGEST_KEYS = ("proposal_id", "tool", "arguments", "workspace")


# --- configuration and the approved spec ----------------------------------------------------------
def load_user_agent(path: Path | None = None) -> str:
    """The operator-chosen User-Agent (Phase 5A.3). Refuses placeholders and generic agents."""
    data = json.loads((path or CONFIG_FILE).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"user_agent"}:
        raise ValueError("wiki_agent_config.json must contain exactly one field: user_agent")
    problem = wiki.user_agent_problem(data["user_agent"])
    if problem:
        raise ValueError(problem)
    return data["user_agent"]


def search_spec_for(user_agent: str) -> dict:
    """Everything about the search except the query. Bound into the approval digest."""
    problem = wiki.user_agent_problem(user_agent, require_real_contact=False)
    if problem:
        raise ValueError(problem)
    return {"provider": wiki.PLAN_CONSTANTS["provider"], "hostname": wiki.HOST, "endpoint": wiki.ENDPOINT,
            "method": "GET", "params": dict(wiki.FIXED_PARAMS), "user_agent": user_agent,
            "max_results": MAX_RESULTS, "timeout_s": TIMEOUT_S, "retries": 0, "redirects": "refused",
            "transmitted": TRANSMITTED}


def plan_for(spec: dict, query: str) -> dict:
    """The Phase 5A.3 adapter plan for one approved query (what the worker receives)."""
    return {**wiki.PLAN_CONSTANTS, "query": query, "user_agent": spec["user_agent"]}


# --- approval binding ------------------------------------------------------------------------------
def proposal_digest(thread_id: str, proposal: dict) -> str:
    """SHA-256 over thread, proposal id, tool, arguments, workspace and (for searches) the search spec.

    For file-tool proposals (no search_spec) this is byte-for-byte the Phase 4C digest.
    """
    core = {k: proposal[k] for k in DIGEST_KEYS}
    if "search_spec" in proposal:
        core["search_spec"] = proposal["search_spec"]
    core["thread_id"] = thread_id
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()


def approval_code(digest: str) -> str:
    return digest[:8].upper()


# --- search records -> graph tool results ---------------------------------------------------------
_ERROR_TEXT = {
    "rate_limited": "Wikipedia rate-limited the request",
    "forbidden": "Wikipedia refused the request (HTTP 403)",
    "unavailable": "Wikipedia was unavailable",
    "server_error": "Wikipedia returned a server error",
    "http_error": "Wikipedia returned an HTTP error",
    "redirect_refused": "Wikipedia answered with a redirect, which is refused",
    "timeout": f"The request timed out after {TIMEOUT_S} seconds",
    "dns_error": "The hostname could not be resolved",
    "tls_error": "The TLS connection could not be verified",
    "connection_error": "The connection failed",
    "malformed_json": "Wikipedia returned a response that is not valid JSON",
    "bad_content_type": "Wikipedia returned an unexpected content type",
    "response_too_large": "Wikipedia's response exceeded the size limit",
    "unexpected_shape": "Wikipedia's response had an unexpected structure",
    "api_error": "Wikipedia's API reported an error",
    "local_rate_limit": "The local search rate limit was reached",
    "broker_refused": "The supervisor refused the search",
    "worker_refused": "The search worker refused the request",
    "worker_error": "The search worker failed",
    "worker_timeout": "The search worker did not finish in time",
}


def _digits(value: object, limit: int = 6) -> str | None:
    s = str(value) if value is not None else ""
    return s if s.isdigit() and len(s) <= limit else None


def error_text(record: dict) -> str:
    status = record.get("status")
    text = _ERROR_TEXT.get(status, "The Wikipedia search failed")
    code, retry = _digits(record.get("http_status"), 3), _digits(record.get("retry_after"))
    if code and "HTTP" not in text:
        text += f" (HTTP {code})"
    if retry:
        text += f"; Retry-After: {retry} s"
    return f"{text}. Stopped safely: no retry was made and no other provider was tried."


def normalize_search_record(record: object, query: str) -> dict:
    """Validate the worker's record before anything reaches the model. Never trusts its URLs."""
    def err(status: str, **extra) -> dict:
        rec = {"status": status, **extra}
        return {"status": "error", "tool": SEARCH_TOOL, "query": query, "text": error_text(rec),
                "structured": None, "search_status": status, "requests": extra.get("requests")}
    if not isinstance(record, dict) or not isinstance(record.get("status"), str):
        return err("worker_error")
    requests = record.get("requests") if isinstance(record.get("requests"), int) else None
    if record["status"] not in ("ok", "empty"):
        status = record["status"] if record["status"] in _ERROR_TEXT else "worker_error"
        return err(status, http_status=record.get("http_status"), retry_after=record.get("retry_after"),
                   requests=requests)
    if record.get("query") != query or requests != 1:
        return err("worker_error", requests=requests)
    items = record.get("results")
    if not isinstance(items, list) or len(items) > MAX_RESULTS:
        return err("unexpected_shape", requests=requests)
    results = []
    for item in items:
        if not (isinstance(item, dict) and isinstance(item.get("title"), str) and isinstance(item.get("url"), str)
                and isinstance(item.get("snippet"), str)):
            return err("unexpected_shape", requests=requests)
        title = wiki.clean_text(item["title"], wiki.MAX_TITLE_CHARS)
        if not title or item["url"] != wiki.article_url(title):   # URLs are rebuilt, never taken on trust
            return err("unexpected_shape", requests=requests)
        results.append({"title": title, "url": wiki.article_url(title),
                        "snippet": wiki.clean_text(item["snippet"], wiki.MAX_SNIPPET_CHARS)})
    return {"status": "success", "tool": SEARCH_TOOL, "query": query, "requests": requests,
            "search_status": "ok" if results else "empty",
            "text": f"{len(results)} Wikipedia result(s)", "structured": {"results": results}}


# --- answer checks -----------------------------------------------------------------------------------
_URLISH = re.compile(r"https?://[^\s<>\"'`]+|(?<![\w/.])(?:[\w-]+\.)*wikipedia\.org/[^\s<>\"'`]*", re.I)
_TRAILING = ".,;:!?)]}'\"*_"


def foreign_urls(text: str, allowed: set[str]) -> list[str]:
    """URLs in the model's text that are not exactly one of the result URLs."""
    bad = []
    for m in _URLISH.finditer(text):
        cand = m.group(0)
        while cand not in allowed and cand and cand[-1] in _TRAILING:
            cand = cand[:-1]
        if cand not in allowed:
            bad.append(m.group(0)[:120])
    return bad


def sources_block(query: str, results: list[dict]) -> str:
    lines = [f'Sources: Wikipedia search for "{query}" ({len(results)} result(s); short search snippets, '
             f"not full articles; {LICENSE_NOTE})"]
    lines += [f"{i}. {r['title']}: {r['url']}" for i, r in enumerate(results, 1)]
    return "\n".join(lines)


def results_listing(results: list[dict], sanitize) -> str:
    return "\n".join(f"{i}. {sanitize(r['title'], 300)}: {sanitize(r['snippet'], 500)}"
                     for i, r in enumerate(results, 1))
