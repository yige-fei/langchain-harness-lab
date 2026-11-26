"""Phase 4B.1: evaluate structured-JSON decision prompts. Proposals are NEVER executed.

Stage 1: every variant on the TUNING set; pick the best by tuning results only.
Stage 2: the baseline and the selected variant on the HELD-OUT set (never used for selection).
Temperature 0 + fixed seed: each (variant, prompt) runs once; repeats would be identical.
"""
import harness_config as cfg  # noqa: I001  (must be first: scrubs environment)

import json
import statistics
import time
from datetime import datetime

import resource_guard
from eval_prompts import FORBIDDEN_IN_SYSTEM, HELD_OUT, TOOL, TUNING, VARIANTS
from experiment_decisions import Decision, path_problem


def judge(item, parsed) -> dict:
    _id, category, prompt, accepted, exp_path, keyword = item
    r = {"schema_valid": parsed is not None}
    if parsed is None:
        return {**r, "correct": False, "error": "schema"}
    action, path, message = parsed.action, parsed.path, parsed.message
    expects_tool = bool(accepted & set(TOOL))
    proposes_tool = action in TOOL
    r["tool_ok"] = (action in accepted) if expects_tool else None
    r["path_ok"] = (path == exp_path) if expects_tool and action in accepted else None
    r["unnecessary_clarification"] = action == "ask_user" and "ask_user" not in accepted
    r["unnecessary_tool"] = proposes_tool and not expects_tool
    r["invented_filename"] = proposes_tool and path not in prompt and f"examples/{path}" not in prompt
    r["unsafe_or_invalid_path"] = proposes_tool and path_problem(action, path) is not None
    ok = action in accepted and (not proposes_tool or path == exp_path)
    if ok and keyword and keyword not in message.lower():
        ok, r["error"] = False, f"answer lacks {keyword!r}"
    if ok and action in ("answer_directly", "ask_user") and not message.strip():
        ok, r["error"] = False, "empty message"
    r["correct"] = ok
    return r


def run(llm, variant: str, system: str, items, stage: str, log) -> list[dict]:
    decider = llm.with_structured_output(Decision, method="json_schema", include_raw=True)
    rows = []
    for item in items:
        resource_guard.check()
        t0 = time.perf_counter()
        out = decider.invoke([("system", system), ("user", item[2])])
        wall = time.perf_counter() - t0
        parsed = out["parsed"]
        row = {"stage": stage, "variant": variant, "id": item[0], "category": item[1], "prompt": item[2],
               "decision": parsed.model_dump() if parsed else None, "raw": out["raw"].content,
               "native_tool_calls": out["raw"].tool_calls, "latency_s": round(wall, 2),
               **judge(item, parsed)}
        rows.append(row)
        log.write(json.dumps(row) + "\n")
        mark = "ok " if row["correct"] else "BAD"
        print(f"  {mark} {variant:17} {item[0]} {item[1]:11} -> {json.dumps(row['decision'])} ({wall:.1f}s)")
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    tool_rows = [r for r in rows if r.get("tool_ok") is not None]
    path_rows = [r for r in rows if r.get("path_ok") is not None]
    lat = sorted(r["latency_s"] for r in rows)
    return {
        "decision_accuracy": f"{sum(r['correct'] for r in rows)}/{n}",
        "tool_selection": f"{sum(r['tool_ok'] for r in tool_rows)}/{len(tool_rows)}",
        "correct_paths": f"{sum(r['path_ok'] for r in path_rows)}/{len(path_rows)}",
        "unnecessary_clarification": sum(r.get("unnecessary_clarification", False) for r in rows),
        "unnecessary_tool_calls": sum(r.get("unnecessary_tool", False) for r in rows),
        "invented_filenames": sum(r.get("invented_filename", False) for r in rows),
        "unsafe_or_invalid_paths": sum(r.get("unsafe_or_invalid_path", False) for r in rows),
        "schema_valid": f"{sum(r['schema_valid'] for r in rows)}/{n}",
        "native_tool_calls_seen": sum(bool(r["native_tool_calls"]) for r in rows),
        "latency_median_s": round(statistics.median(lat), 2),
        "latency_max_s": lat[-1],
    }


def main() -> None:
    for name, text in VARIANTS.items():
        if name != "B0-baseline-4B":
            assert not any(f in text for f in FORBIDDEN_IN_SYSTEM), f"{name} contains an example file name"
    llm = cfg.make_llm()
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    out = cfg.RESULTS_DIR / f"eval_4b1_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    start, mem0, disk0 = time.perf_counter(), resource_guard.memory_pressure_level(), resource_guard.free_gib()
    with out.open("w") as log:
        print("== Stage 1: tuning set")
        tuning = {v: summarize(run(llm, v, s, TUNING, "tuning", log)) for v, s in VARIANTS.items()}
        candidates = [v for v in VARIANTS if v != "B0-baseline-4B"]

        def rank(v):
            s = tuning[v]
            return (int(s["decision_accuracy"].split("/")[0]), -s["unsafe_or_invalid_paths"],
                    -s["invented_filenames"], -s["unnecessary_tool_calls"] - s["unnecessary_clarification"])
        best = max(candidates, key=rank)
        print(f"== Selected on tuning set only: {best}")
        print("== Stage 2: held-out set")
        held = {v: summarize(run(llm, v, VARIANTS[v], HELD_OUT, "held_out", log)) for v in ("B0-baseline-4B", best)}
        log.write(json.dumps({"summary": {"tuning": tuning, "held_out": held, "selected": best}}) + "\n")
    print("\n== Tuning-set summary");  [print(f"{v:17} {json.dumps(s)}") for v, s in tuning.items()]
    print("== Held-out summary");    [print(f"{v:17} {json.dumps(s)}") for v, s in held.items()]
    print(f"model calls: {len(VARIANTS) * len(TUNING) + 2 * len(HELD_OUT)}  wall: {time.perf_counter() - start:.0f}s")
    print(f"disk free {disk0:.2f} -> {resource_guard.free_gib():.2f} GiB; pressure level {mem0} -> "
          f"{resource_guard.memory_pressure_level()}; log: {out}")


if __name__ == "__main__":
    main()
