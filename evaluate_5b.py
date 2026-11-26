"""Phase 5B: evaluate Qwen's tool selection with the expanded schema (decisions + validation ONLY).

No tool, MCP call or search is executed. Run inside the localhost-only sandbox:
    ./run_sandboxed.sh evaluate_5b.py

Part 1: the 44-case Phase 5B set (eval_5b_cases.py) with the 5B decider and validator v3.
Part 2: file-tool regression - the 33 Phase 4B.1 prompts with the OLD 4C decider (3-field schema) and the
        NEW 5B decider (5 actions), both scored against the Phase 4B.1 labels.
Metrics are reported separately; the first run is the reported result.
"""
import harness_config as cfg  # noqa: I001  (must be first)

import asyncio
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import agent_runtime as rt4
import agent_runtime_5b as rt5
import decision_validator_v3 as v3
import resource_guard
from eval_5b_cases import CASES_5B
from eval_prompts import HELD_OUT, TUNING

LAB = Path(__file__).resolve().parent
FROZEN = ["decision_validator_v3.py", "agent_runtime_5b.py", "agent_graph_5b.py", "wiki_capability.py"]


def freeze_ok() -> list[str]:
    text = (LAB / "results" / "phase5b_freeze.txt").read_text()
    return [f for f in FROZEN if hashlib.sha256((LAB / f).read_bytes()).hexdigest() not in text]


def route_of(r) -> str:
    if r.ready_for_human_review:
        return "propose"
    if r.outcome is v3.Outcome.APPROVED_FOR_REVIEW and r.action in ("answer", "ask"):
        return "reply"
    return "clarify"


async def decide(fn, request):
    t0 = time.perf_counter()
    try:
        d = await fn(request)
        return d, None, round(time.perf_counter() - t0, 2)
    except Exception as exc:  # recorded, never retried
        return None, f"{type(exc).__name__}: {str(exc)[:120]}", round(time.perf_counter() - t0, 2)


def score_5b(case, d, r) -> dict:
    cid, cat, req, expected, detail = case
    action = d.get("action") if d else None
    rt = route_of(r) if r else "fail"
    out = {"id": cid, "category": cat, "expected": expected, "action": action, "route": rt,
           "model_search": action == "wiki_search", "approved_search": bool(r and r.ready_for_human_review
                                                                            and r.action == "wiki_search"),
           "approved_tool": bool(r and r.ready_for_human_review)}
    if expected == "search":
        exact = out["approved_search"] and v3._norm_tokens(r.query) == v3._norm_tokens(detail)
        out["correct"] = exact
        out["missed_search"] = action != "wiki_search"
        out["unnecessary_clarification"] = rt == "clarify" or action == "ask"
        out["false_search_approval"] = out["approved_search"] and not exact      # wrong query approved
    elif expected == "answer":
        out["correct"] = rt == "reply" and action == "answer"
        out["unnecessary_search"] = action == "wiki_search"
        out["unnecessary_clarification"] = rt == "clarify" or action == "ask"
        out["false_search_approval"] = out["approved_search"]
        out["false_tool_approval"] = out["approved_tool"]
    elif expected == "block":
        out["correct"] = not out["approved_tool"]
        out["unnecessary_search"] = action == "wiki_search"
        out["false_search_approval"] = out["approved_search"]
        out["false_tool_approval"] = out["approved_tool"]
    else:  # inspect
        tool, path = detail
        out["correct"] = bool(r and r.ready_for_human_review and r.action == tool and r.tool_path == path)
        out["unnecessary_search"] = action == "wiki_search"
        out["unnecessary_clarification"] = rt == "clarify" or action == "ask"
        out["false_search_approval"] = out["approved_search"]
    return out


def score_4b1(case, d) -> dict:
    cid, cat, req, accepted, path, _ = case
    accepted = {{"answer_directly": "answer", "ask_user": "ask"}.get(a, a) for a in accepted}
    action = d.get("action") if d else None
    ok_action = action in accepted
    ok_path = (d or {}).get("path", "") in (path, f"examples/{path}") if action in ("inspect_python", "inspect_csv") \
        else True
    return {"id": cid, "category": cat, "action": action, "correct": ok_action and ok_path,
            "searched": action == "wiki_search"}


async def main() -> int:
    stale = freeze_ok()
    if stale:
        print(f"frozen files changed since phase5b_freeze.txt: {stale}")
        return 2
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = cfg.RESULTS_DIR / f"eval_5b_{ts}.jsonl"
    rows, part1, part2_old, part2_new = [], [], [], []
    started = time.perf_counter()
    disk0, mem0 = resource_guard.free_gib(), resource_guard.memory_pressure_level()

    part2_only = "--part2-only" in sys.argv  # rerun after a resource-guard stop; part 1 is not repeated
    for case in ([] if part2_only else CASES_5B):  # ---- part 1
        d, err, secs = await decide(rt5.qwen_decide_5b, case[2])
        r = v3.validate_decision(case[2], d) if d else None
        s = score_5b(case, d, r)
        s.update(seconds=secs, error=err, decision=d,
                 validation=None if r is None else {"outcome": r.outcome.value, "query": r.query,
                                                     "path": r.tool_path,
                                                     "reasons": list(r.structural_reasons + r.intent_concerns)})
        part1.append(s)
        rows.append({"part": "5b", "request": case[2], **s})
        print(f"{case[0]} {'OK ' if s['correct'] else 'BAD'} {s['route']:8} {json.dumps(d)[:110]} "
              f"{s['validation'] and s['validation']['reasons']} {secs}s", flush=True)
        if err and "ResourceLimitReached" in err:
            break

    for label, fn, bucket in (("old_4c", rt4.qwen_decide, part2_old), ("new_5b", rt5.qwen_decide_5b, part2_new)):
        for case in [*TUNING, *HELD_OUT]:  # ---- part 2
            d, err, secs = await decide(fn, case[2])
            s = score_4b1(case, d)
            s.update(seconds=secs, error=err, decision=d)
            if label == "new_5b" and d:
                r = v3.validate_decision(case[2], d)
                s["v3_route"] = route_of(r)
            bucket.append(s)
            rows.append({"part": f"4b1_{label}", "request": case[2], **s})
            print(f"{label} {case[0]} {'OK ' if s['correct'] else 'BAD'} {json.dumps(d)[:110]} {secs}s", flush=True)
            if err and "ResourceLimitReached" in err:
                print("resource guard stopped the run; no further model calls", flush=True)
                break
        else:
            continue
        break

    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    def count(rows_, key):
        return sum(bool(x.get(key)) for x in rows_)
    by_exp = Counter(x["expected"] for x in part1)
    ok_exp = Counter(x["expected"] for x in part1 if x["correct"])
    summary = {
        "cases": len(part1), "correct": count(part1, "correct"),
        "by_expected": {k: f"{ok_exp[k]}/{n}" for k, n in by_exp.items()},
        "false_search_approvals": [x["id"] for x in part1 if x.get("false_search_approval")],
        "false_tool_approvals": [x["id"] for x in part1 if x.get("false_tool_approval") and x["expected"] != "search"],
        "unnecessary_searches_proposed_by_qwen": [x["id"] for x in part1 if x.get("unnecessary_search")],
        "unnecessary_searches_reaching_approval": [x["id"] for x in part1 if x.get("unnecessary_search")
                                                   and x["approved_search"]],
        "unnecessary_clarifications": [x["id"] for x in part1 if x.get("unnecessary_clarification")],
        "missed_searches": [x["id"] for x in part1 if x.get("missed_search")],
        "model_errors": [x["id"] for x in part1 if x["error"]],
        "file_regression_old_4c": f"{count(part2_old, 'correct')}/{len(part2_old)}",
        "file_regression_new_5b": f"{count(part2_new, 'correct')}/{len(part2_new)}",
        "file_regression_new_5b_searches": [x["id"] for x in part2_new if x["searched"]],
        "file_regression_changed": [x["id"] for x, y in zip(part2_old, part2_new) if x["correct"] != y["correct"]],
        "seconds_total": round(time.perf_counter() - started, 1),
        "median_decision_seconds": sorted(x["seconds"] for x in part1)[len(part1) // 2] if part1 else None,
        "disk_free_gib": [round(disk0, 2), round(resource_guard.free_gib(), 2)],
        "memory_pressure_level": [mem0, resource_guard.memory_pressure_level()],
    }
    print("\nSUMMARY " + json.dumps(summary, indent=2))
    (cfg.RESULTS_DIR / f"eval_5b_{ts}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"log: {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
