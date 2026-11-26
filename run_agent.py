"""Phase 4C command-line runner: one request, at most one approved tool call.

Approval modes (none of them can approve by itself):
  --approval tty   (default) you type the shown approval code in this terminal; anything else denies.
  --approval fifo  read ONE line from a named pipe: "approve <CODE>" or anything else (deny). Used to
                   relay a decision a human made in another UI; this program never writes to the pipe.
  --approval deny  every proposed tool call is denied (batch/no-execution runs).

Run inside the localhost-only sandbox:  ./run_sandboxed.sh run_agent.py "What functions are in demo.py?"
"""
import harness_config as cfg  # noqa: I001  (must be first)

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime, timezone

import agent_graph as ag
import agent_runtime as rt
import resource_guard


def show_card(card: dict) -> None:
    print("\n=================== APPROVAL REQUIRED ===================")
    print(f"tool        : {card['tool']}  (read-only SamToolkit MCP tool, server.py over stdio)")
    print(f"arguments   : {json.dumps(card['arguments'])}")
    print(f"workspace   : {card['workspace']}")
    print(f"user wrote  : {card['user_reference']}")
    print(f"request     : {card['request']}")
    print(f"proposal id : {card['proposal_id']}")
    print(f"APPROVAL CODE: {card['approval_code']}")
    print("==========================================================", flush=True)


def collect_decision(card: dict, mode: str, fifo: str | None) -> dict:
    code = card["approval_code"]
    if mode == "deny":
        print("approval mode 'deny': denied automatically; nothing will run.")
        return ag.human_response(card, False, via="auto-deny")
    if mode == "tty":
        if not sys.stdin.isatty():
            print("stdin is not an interactive terminal: denied.")
            return ag.human_response(card, False, via="tty-unavailable")
        typed = input(f"Type {code} to approve this exact call (anything else denies): ").strip()
        return ag.human_response(card, typed == code, via="tty")
    print(f"Waiting for a human decision relayed through {fifo} ...", flush=True)
    with open(fifo, encoding="utf-8") as f:
        line = f.readline().strip()
    approved = line == f"approve {code}"
    print(f"relayed decision received: {'APPROVE' if approved else 'DENY'} ({line[:40]!r})", flush=True)
    return ag.human_response(card, approved, via="relayed-human-dialog")


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("request")
    p.add_argument("--approval", choices=("tty", "fifo", "deny"), default="tty")
    p.add_argument("--fifo")
    args = p.parse_args()
    if args.approval == "fifo" and not args.fifo:
        p.error("--fifo is required with --approval fifo")

    thread_id = f"cli-{uuid.uuid4().hex[:12]}"
    graph = ag.build_graph(rt.qwen_decide, rt.mcp_execute, rt.qwen_answer, str(cfg.WORKSPACE))
    disk0, mem0, t0 = resource_guard.free_gib(), resource_guard.memory_pressure_level(), time.perf_counter()
    print(f"thread {thread_id} | request: {args.request!r}")
    out = await ag.start(graph, args.request, thread_id)
    values = (await graph.aget_state(ag.config_for(thread_id))).values
    print(f"model decision : {json.dumps(values.get('decision'))}")
    print(f"validation     : {json.dumps(values.get('validation'))}")
    t_decide = time.perf_counter() - t0
    if out["status"] == "awaiting_approval":
        show_card(out["card"])
        response = collect_decision(out["card"], args.approval, args.fifo)
        t1 = time.perf_counter()
        out = await ag.resume(graph, thread_id, response)
        t_after = time.perf_counter() - t1
    else:
        t_after = 0.0
    print("\n----- FINAL -----\n" + out["final"] + "\n-----------------")
    record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "thread_id": thread_id,
              "request": args.request, "approval_mode": args.approval, "events": out.get("events"),
              "tool_calls_made": out.get("tool_calls_made"), "mcp_executions": rt.MCP_EXECUTIONS,
              "decision": out.get("decision"), "validation": out.get("validation"), "final": out["final"],
              "seconds_to_decision": round(t_decide, 2), "seconds_after_approval": round(t_after, 2),
              "disk_free_gib": [round(disk0, 2), round(resource_guard.free_gib(), 2)],
              "memory_pressure_level": [mem0, resource_guard.memory_pressure_level()]}
    print("audit:", json.dumps({k: record[k] for k in ("events", "tool_calls_made", "mcp_executions",
                                                       "seconds_to_decision", "seconds_after_approval",
                                                       "disk_free_gib", "memory_pressure_level")}))
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    with (cfg.RESULTS_DIR / "agent_runs.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
