"""Phase 5B search worker: runs ONE approved Wikipedia search inside the search sandbox.

Started only by supervisor_5b.py, once per approved search, as
    sandbox-exec -p <HTTPS 443 + system resolver profile> python -I search_lab/wiki_worker.py
with a minimal environment. It reads one JSON object from stdin,
    {"plan": <Phase 5A.3 plan>, "approval_code": <code of that exact plan>}
re-validates it, sends exactly one request through the verified Phase 5A.3 adapter (no retries,
no redirects, no proxies) and prints one JSON record on stdout. No LLM, MCP or LangGraph.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))  # appended after the standard library: cannot shadow it
import wikipedia_search as ws  # noqa: E402

MAX_INPUT_BYTES = 16_384


def validate_plan(plan: object) -> str | None:
    if not isinstance(plan, dict) or set(plan) != {*ws.PLAN_CONSTANTS, "query", "user_agent"}:
        return "plan fields differ from the approved schema"
    for key, value in ws.PLAN_CONSTANTS.items():
        if plan[key] != value:
            return f"plan field {key} differs from the fixed value"
    return ws.query_problem(plan["query"]) or ws.user_agent_problem(plan["user_agent"])


def run(raw: str, opener=None) -> dict:
    refused = {"status": "worker_refused", "requests": 0, "results": []}
    if len(raw) > MAX_INPUT_BYTES:
        return {**refused, "error": "input too large"}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {**refused, "error": "input is not JSON"}
    if not isinstance(data, dict) or set(data) != {"plan", "approval_code"}:
        return {**refused, "error": "unexpected input fields"}
    problem = validate_plan(data["plan"])
    if problem:
        return {**refused, "error": problem}
    try:
        return ws.run_plan(data["plan"], str(data["approval_code"]), opener)
    except (PermissionError, ValueError) as exc:
        return {**refused, "error": ws.clean_text(exc, 120)}


def main() -> int:
    record = run(sys.stdin.read(MAX_INPUT_BYTES + 1))
    sys.stdout.write(json.dumps(record) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
