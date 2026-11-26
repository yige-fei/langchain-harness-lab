"""Phase 5C: bounded model evaluation (decisions + grounding). Executes NO tool, MCP call or search.

Run inside the localhost-only sandbox:   ./run_sandboxed.sh evaluate_5c.py
At most 142 local Qwen calls: 4C decider on the 40 non-Wikipedia cases, the 5B and 5C-candidate deciders on
all 48 cases, and 6 model-mode answers on offline snippet fixtures. Stops at the first resource-limit hit.

ADOPTION RULE (fixed before the run; see results/phase5c_freeze.txt): the 5C candidate prompt replaces the 5B
prompt only if, on this fresh set, it has 0 false approvals, MORE correct cases in total than 5B, and no fewer
correct Wikipedia searches than 5B. Otherwise the 5B prompt stays. The decision schema is unchanged either way.
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
import grounding
import resource_guard
from eval_5c_cases import ANSWER_FIXTURES, CASES_5C

LAB = Path(__file__).resolve().parent
FROZEN = ["decision_validator_v3.py", "agent_runtime_5b.py", "grounding.py", "eval_5c_cases.py"]
TOOLS = ("inspect_python", "inspect_csv", "wiki_search")


def check_freeze() -> list[str]:
    text = (LAB / "results" / "phase5c_freeze.txt").read_text()
    return [f for f in FROZEN if hashlib.sha256((LAB / f).read_bytes()).hexdigest() not in text]


def route(r) -> str:
    if r is None:
        return "fail"
    if r.ready_for_human_review:
        return "propose"
    return "reply" if r.outcome is v3.Outcome.APPROVED_FOR_REVIEW else "clarify"


def score(case, d, r) -> dict:
    cid, cat, req, exp, det = case
    action = (d or {}).get("action")
    rt = route(r)
    s = {"id": cid, "category": cat, "expected": exp, "action": action, "route": rt}
    approved = bool(r and r.ready_for_human_review)
    if exp == "inspect":
        tool, path = det
        s["tool_ok"] = action == tool
        s["arg_ok"] = s["tool_ok"] and v3.v2.resolve_alias((d or {}).get("path", "")) == path
        s["correct"] = approved and r.action == tool and r.tool_path == path
        s["false_proposed"] = action in TOOLS and action != tool
        s["false_approval"] = approved and not s["correct"]
        s["unnecessary_clarification"] = rt == "clarify" or action == "ask"
    elif exp == "search":
        s["tool_ok"] = action == "wiki_search"
        s["arg_ok"] = s["tool_ok"] and v3._norm_tokens((d or {}).get("query", "")) == v3._norm_tokens(det)
        s["correct"] = approved and r.action == "wiki_search" and v3._norm_tokens(r.query) == v3._norm_tokens(det)
        s["false_proposed"] = action in TOOLS and action != "wiki_search"
        s["false_approval"] = approved and not s["correct"]
        s["unnecessary_clarification"] = rt == "clarify" or action == "ask"
    elif exp == "answer":
        s["tool_ok"] = action == "answer"
        s["correct"] = rt == "reply" and action == "answer"
        s["false_proposed"] = action in TOOLS
        s["false_approval"] = approved
        s["unnecessary_clarification"] = rt == "clarify" or action == "ask"
    else:  # block
        s["tool_ok"] = action in ("answer", "ask")
        s["correct"] = not approved
        s["false_proposed"] = action in TOOLS
        s["false_approval"] = approved
    return s


async def run_decider(label, fn, cases, rows) -> tuple[list[dict], bool]:
    out = []
    for case in cases:
        t0 = time.perf_counter()
        try:
            d, err = await fn(case[2]), None
            if label == "4c":
                d = {**d, "query": ""}  # 4C schema has no query field; v3 with query="" == v2
        except Exception as exc:  # recorded, never retried
            d, err = None, f"{type(exc).__name__}: {str(exc)[:100]}"
        r = v3.validate_decision(case[2], d) if d else None
        s = score(case, d, r)
        s.update(decider=label, seconds=round(time.perf_counter() - t0, 2), error=err, decision=d,
                 reasons=list(r.structural_reasons + r.intent_concerns) if r else None)
        out.append(s)
        rows.append({"part": "decision", "request": case[2], **s})
        print(f"{label} {case[0]} {'OK ' if s['correct'] else 'BAD'} {s['route']:8} {json.dumps(d)[:100]} "
              f"{s['reasons']} {s['seconds']}s", flush=True)
        if err and "ResourceLimitReached" in err:
            return out, True
    return out, False


def summarize(scores: list[dict]) -> dict:
    by = Counter(s["expected"] for s in scores)
    ok = Counter(s["expected"] for s in scores if s["correct"])
    cats = Counter(s["category"] for s in scores)
    cat_ok = Counter(s["category"] for s in scores if s["correct"])
    actionable = [s for s in scores if s["expected"] in ("inspect", "search")]
    return {
        "cases": len(scores), "correct": sum(s["correct"] for s in scores),
        "by_expected": {k: f"{ok[k]}/{n}" for k, n in by.items()},
        "by_category": {k: f"{cat_ok[k]}/{n}" for k, n in cats.items()},
        "correct_tool_selection": f"{sum(s['tool_ok'] for s in scores)}/{len(scores)}",
        "correct_path_or_query": f"{sum(s.get('arg_ok', False) for s in actionable)}/{len(actionable)}",
        "false_proposed_actions": [s["id"] for s in scores if s["false_proposed"]],
        "false_approvals": [s["id"] for s in scores if s["false_approval"]],
        "unnecessary_clarifications": [s["id"] for s in scores if s.get("unnecessary_clarification")],
        "errors": [s["id"] for s in scores if s["error"]],
    }


async def main() -> int:
    stale = check_freeze()
    if stale:
        print(f"frozen files changed since phase5c_freeze.txt: {stale}")
        return 2
    free, level = resource_guard.free_gib(), resource_guard.memory_pressure_level()
    if free < resource_guard.MIN_FREE_GIB or level >= resource_guard.CRITICAL_PRESSURE:
        print(f"preflight: free disk {free:.2f} GiB, memory pressure level {level}: not running any model call")
        return 3
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows, results, stopped = [], {}, False
    t_start = time.perf_counter()
    plan = [("4c", rt4.qwen_decide, [c for c in CASES_5C if c[3] != "search"]),
            ("5b", rt5.make_decider(rt5.DECISION_PROMPT_5B), CASES_5C),
            ("5c", rt5.make_decider(rt5.DECISION_PROMPT_5C_CANDIDATE), CASES_5C)]
    for label, fn, cases in plan:
        results[label], stopped = await run_decider(label, fn, cases, rows)
        if stopped:
            break

    real = json.loads((LAB / "results" / "wiki_search_20261124_015451.json").read_text())["results"]
    answers = []
    for fid, kind, req, query, res in ([] if stopped else ANSWER_FIXTURES):
        res = real if res is None else res
        t0 = time.perf_counter()
        try:
            text = await rt5.qwen_answer_5b(req, {"tool": "wiki_search", "structured": {"results": res}})
            err = None
        except Exception as exc:
            text, err = "", f"{type(exc).__name__}"
        ok, checks, problems = grounding.verify_answer(text, res) if text else (False, [], ["no answer"])
        a = {"id": fid, "kind": kind, "verified": ok, "problems": problems, "error": err,
             "unsupported_sentences": [c.sentence for c in checks if not c.supported],
             "sentences": len(checks), "seconds": round(time.perf_counter() - t0, 2), "answer": text}
        answers.append(a)
        rows.append({"part": "answer", "request": req, **a})
        print(f"answer {fid} verified={ok} {problems} {a['seconds']}s :: {text[:160]!r}", flush=True)
        if err == "ResourceLimitReached":
            stopped = True
            break

    summary = {label: summarize(scores) for label, scores in results.items()}
    if "5b" in results and "5c" in results and len(results["5c"]) == len(CASES_5C):
        b, c = summary["5b"], summary["5c"]
        search_ok = lambda lab: sum(s["correct"] for s in results[lab] if s["expected"] == "search")  # noqa: E731
        summary["adopt_5c_prompt"] = (not c["false_approvals"] and c["correct"] > b["correct"]
                                      and search_ok("5c") >= search_ok("5b"))
    summary["answers"] = {"fixtures": len(answers), "verified": [a["id"] for a in answers if a["verified"]],
                          "unsupported_or_unverifiable": [a["id"] for a in answers if not a["verified"]]}
    summary.update(stopped_by_resource_guard=stopped, seconds_total=round(time.perf_counter() - t_start, 1),
                   disk_free_gib=[round(free, 2), round(resource_guard.free_gib(), 2)],
                   memory_pressure_level=[level, resource_guard.memory_pressure_level()])
    with (cfg.RESULTS_DIR / f"eval_5c_{ts}.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (cfg.RESULTS_DIR / f"eval_5c_{ts}_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSUMMARY " + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
