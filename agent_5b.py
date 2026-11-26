"""Phase 5B agent process. Started ONLY by supervisor_5b.py, inside the localhost-only sandbox.

It runs one request through the Phase 5B LangGraph workflow. Approval cards go to the supervisor
(which shows them and collects the human decision); an approved search is handed to the supervisor,
which runs the separately sandboxed worker. stdin is /dev/null: this process cannot read the terminal.
"""
import harness_config as cfg  # noqa: I001  (must be first)

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime, timezone

import agent_graph_5b as g5
import agent_runtime_5b as rt
import broker_protocol as bp
import resource_guard
import wiki_capability as wc


REPORT_LIMIT = 200_000  # below broker_protocol.MAX_MESSAGE_BYTES


def _clip(value, limit: int):
    text = json.dumps(value, ensure_ascii=True, default=str)
    return value if len(text) <= limit else {"omitted": f"{len(text)} bytes"}


def build_report(record: dict, values: dict) -> dict:
    """Phase 6B: the structured trace sent to the supervisor with "done". Everything in it is AGENT-REPORTED
    (the supervisor and the UI label it so); it holds no credentials, environment data or absolute paths
    beyond the workspace label the human already approves against."""
    proposal = values.get("proposal") or None
    approval = values.get("approval") or None
    result = values.get("tool_result") or None
    report = {k: record.get(k) for k in ("thread_id", "request", "answer_mode", "decision", "validation", "events",
                                         "final", "tool_calls_made", "seconds_to_decision", "seconds_after_approval",
                                         "disk_free_gib", "memory_pressure_level")}
    report["mcp_executions"] = len(record.get("mcp_executions") or [])
    report["search_handoffs"] = len(record.get("search_handoffs") or [])
    report["proposal"] = ({k: proposal.get(k) for k in ("proposal_id", "tool", "arguments", "workspace", "digest")}
                          if isinstance(proposal, dict) else None)
    report["approval"] = ({k: approval.get(k) for k in ("granted", "via", "proposal_id")}
                          if isinstance(approval, dict) else None)
    if isinstance(result, dict):
        report["tool_result"] = {k: result.get(k) for k in ("status", "tool", "text", "search_status", "exception")
                                 if k in result}
        report["tool_result"]["structured"] = _clip(result.get("structured"), 120_000)
    else:
        report["tool_result"] = None
    report["final"] = (report["final"] or "")[:12_000]
    if len(json.dumps(report, ensure_ascii=True, default=str)) > REPORT_LIMIT and report["tool_result"]:
        report["tool_result"]["structured"] = {"omitted": "too large"}
    return report


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--broker-in", type=int, required=True)
    p.add_argument("--broker-out", type=int, required=True)
    p.add_argument("--answer-mode", choices=("evidence", "model"), default="evidence")
    p.add_argument("request")
    args = p.parse_args()

    broker = bp.BrokerClient(args.broker_in, args.broker_out)
    spec = wc.search_spec_for(wc.load_user_agent())
    graph = g5.build_graph(rt.qwen_decide_5b, rt.make_execute(broker), rt.qwen_answer_5b, str(cfg.WORKSPACE), spec,
                           answer_mode=args.answer_mode)
    thread_id = f"5b-{uuid.uuid4().hex[:12]}"
    disk0, mem0, t0 = resource_guard.free_gib(), resource_guard.memory_pressure_level(), time.perf_counter()
    print(f"[agent] thread {thread_id} | request: {args.request!r}", flush=True)
    out = await g5.start(graph, args.request, thread_id)
    values = (await graph.aget_state(g5.config_for(thread_id))).values
    print(f"[agent] model decision : {json.dumps(values.get('decision'))}", flush=True)
    print(f"[agent] validation     : {json.dumps(values.get('validation'))}", flush=True)
    t_decide, t_after = time.perf_counter() - t0, 0.0
    if out["status"] == "awaiting_approval":
        print("[agent] LangGraph paused at the approval interrupt; card sent to the supervisor.", flush=True)
        card = out["card"]
        try:
            reply = await asyncio.to_thread(broker.call, {"type": "approval_request", "card": card},
                                            "approval_response")
            response = {k: reply.get(k) for k in ("decision", "proposal_id", "digest", "via")}
        except (EOFError, bp.ProtocolError, OSError):
            response = {"decision": "deny", "via": "broker_unavailable"}
        t1 = time.perf_counter()
        out = await g5.resume(graph, thread_id, response)
        t_after = time.perf_counter() - t1
    print("\n----- FINAL -----\n" + out["final"] + "\n-----------------", flush=True)
    record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "thread_id": thread_id,
              "request": args.request, "answer_mode": args.answer_mode, "events": out.get("events"), "tool_calls_made": out.get("tool_calls_made"),
              "mcp_executions": rt.MCP_EXECUTIONS, "search_handoffs": rt.SEARCH_REQUESTS,
              "decision": out.get("decision"), "validation": out.get("validation"), "final": out["final"],
              "seconds_to_decision": round(t_decide, 2), "seconds_after_approval": round(t_after, 2),
              "disk_free_gib": [round(disk0, 2), round(resource_guard.free_gib(), 2)],
              "memory_pressure_level": [mem0, resource_guard.memory_pressure_level()]}
    print("[agent] audit:", json.dumps({k: record[k] for k in ("events", "tool_calls_made", "mcp_executions",
                                                               "search_handoffs", "seconds_to_decision",
                                                               "seconds_after_approval", "disk_free_gib",
                                                               "memory_pressure_level")}), flush=True)
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    with (cfg.RESULTS_DIR / "agent_runs_5b.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    final_values = (await graph.aget_state(g5.config_for(thread_id))).values
    report = build_report(record, final_values)
    try:
        broker.notify({"type": "done", "tool_calls_made": out.get("tool_calls_made"), "report": report})
    except (OSError, bp.ProtocolError):
        try:  # too large even after trimming: send the essentials only
            broker.notify({"type": "done", "tool_calls_made": out.get("tool_calls_made"),
                           "report": {k: report.get(k) for k in ("thread_id", "events", "tool_calls_made",
                                                                 "mcp_executions", "search_handoffs")}})
        except (OSError, bp.ProtocolError):
            pass
    broker.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
