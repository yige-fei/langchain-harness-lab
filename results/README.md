# results/

The harness writes its raw run records here: the supervisor's and agent's run logs, evaluation
records, isolation-check reports and the local Wikipedia rate-limit state. Those raw files are
**not published**. They contain request text, model output, historical approval codes and readings
from the machine they ran on (free disk, memory pressure). `.gitignore` excludes everything in this
directory except the files listed below.

The directory itself must exist. The sandboxed agent may write only its own log file
(`agent_runs_5b.jsonl`), not create this directory, so a checkout without `results/` would fail
on its first agent run.

## Published files and why

| File | Why it is needed | Review notes |
|---|---|---|
| `wiki_search_20261124_015451.json` | Test fixture: `tests/test_grounding.py` and `evaluate_5c.py` read its `results` list (real Wikipedia search snippets from Phase 5A.3). | One public Wikipedia API response: 5 titles, article URLs and short snippets. No personal data. **Sanitized for publication:** the original record's `plan_sha256` field was removed, because its first 8 hex digits are a historical approval code. Nothing reads that field. Every other field, including `results`, is unchanged, and the untouched original is kept locally (ignored) as `wiki_search_20261124_015451.raw.json`. |
| `phase6d_freeze.txt` | Read by `evaluate_6d.py`, which refuses a model run if any frozen file changed. | File hashes and a timestamp only. **Matches the current code.** |
| `phase5c_freeze.txt` | Read by `evaluate_5c.py`. | File hashes and a timestamp only. **Historical:** `agent_graph_5b.py` and `agent_runtime_5b.py` changed in later phases, so `evaluate_5c.py` now reports them as changed. |
| `phase5b_freeze.txt` | Read by `evaluate_5b.py`. | File hashes and a timestamp only. **Historical:** `decision_validator_v3.py`, `agent_runtime_5b.py` and `agent_graph_5b.py` changed in later phases. |
| `eval_6d_reviewed_summary.json` | Evidence for the Phase 6D figures quoted in the README. | Aggregate counts and case IDs only (no prompts, no model text). |

Earlier phases' figures (4B–5C) are not republished here. Their raw records stay local.

## Third-party content

The snippets in `wiki_search_20261124_015451.json` are text from Wikipedia articles by Wikipedia
contributors, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Each snippet keeps its source article URL. This repository's MIT licence does not apply to that
text.
