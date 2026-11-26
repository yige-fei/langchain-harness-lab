"""Phase 4B.2 offline evaluation of decision_validator. No model, MCP or network.

1. FRESH set (fresh_eval_cases.py), written after the validator was frozen: the
   independent result. Run once; the first run is the one reported.
2. RETROSPECTIVE replay of the 92 recorded Phase 4B.1 model decisions. These
   prompts informed the validator's design, so this is NOT an independent score.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from decision_validator import validate_decision
from fresh_eval_cases import FRESH_CASES

LAB = Path(__file__).resolve().parent
LEGACY_ACTION_NAMES = {"answer_directly": "answer", "ask_user": "ask"}  # explicit, documented mapping


def sha(path: str) -> str:
    return hashlib.sha256((LAB / path).read_bytes()).hexdigest()


def fresh() -> None:
    print("== FRESH SET (independent)")
    print("validator sha256:", sha("decision_validator.py"))
    print("fresh set sha256:", sha("fresh_eval_cases.py"))
    counts = Counter()
    errors = []
    for cid, user, decision, expected, note in FRESH_CASES:
        r = validate_decision(user, decision)
        got = "approve" if r.approved else "reject"
        counts[(expected, got)] += 1
        flag = "ok " if got == expected else ("FALSE APPROVAL" if got == "approve" else "FALSE REJECTION")
        print(f"  {flag:15} {cid} expected={expected:7} got={got:7} reasons={[x.value for x in r.reasons]}  [{note}]")
        if got != expected:
            errors.append(cid)
    n = len(FRESH_CASES)
    print(f"  cases={n} correct={counts[('approve','approve')] + counts[('reject','reject')]}/{n}")
    print(f"  should-approve: approved {counts[('approve','approve')]}, FALSE REJECTIONS {counts[('approve','reject')]}")
    print(f"  should-reject : rejected {counts[('reject','reject')]}, FALSE APPROVALS {counts[('reject','approve')]}")
    print(f"  misclassified: {errors}")


def retrospective(log: Path) -> None:
    print(f"\n== RETROSPECTIVE replay of Phase 4B.1 decisions (NOT independent): {log.name}")
    rows = [json.loads(line) for line in log.open() if '"summary"' not in line]
    table = Counter()
    examples = {}
    for r in rows:
        dec = r["decision"]
        dec = {**dec, "action": LEGACY_ACTION_NAMES.get(dec["action"], dec["action"])}
        v = validate_decision(r["prompt"], dec)
        is_tool = dec["action"] in ("inspect_python", "inspect_csv")
        key = ("tool" if is_tool else "non-tool", "4B.1-correct" if r["correct"] else "4B.1-wrong",
               "approved" if v.approved else "rejected")
        table[key] += 1
        examples.setdefault(key, []).append(f'{r["variant"]}/{r["id"]}: {r["prompt"]!r} -> {json.dumps(dec)} '
                                            f'{[x.value for x in v.reasons]}')
    for key in sorted(table):
        print(f"  {key}: {table[key]}")
    for key in sorted(examples):
        if key[1:] in (("4B.1-wrong", "approved"), ("4B.1-correct", "rejected")):
            print(f"  -- {key}")
            for e in examples[key]:
                print("     ", e)


if __name__ == "__main__":
    fresh()
    logs = sorted((LAB / "results").glob("eval_4b1_*.jsonl"))
    if logs and "--fresh-only" not in sys.argv:
        retrospective(logs[-1])
