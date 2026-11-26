# langchain-harness-lab

A supervised, local LangGraph agent harness. It lets the local Qwen2.5-Coder 3B model (via Ollama) propose
**one** read-only action per request. A human must approve that action before anything runs. The possible
actions are:

- one [SamToolkit](https://github.com/yige-fei/sam-toolkit-mcp) file check (`inspect_python` / `inspect_csv`
  over MCP), or
- one Wikipedia search.

[LocalCodePilot](https://github.com/yige-fei/Local-Code-Pilot) uses it as its **AI Agent** mode.

**Status:** a personal research harness, current to Phase 6D. It is macOS-only, tied to a fixed directory
layout, and has been tested on one machine. It is not a general-purpose agent framework, and a fresh
install has not been verified (see [Limitations](#limitations)).

## Required layout (this release)

The paths are hardcoded, so the three repositories must be cloned exactly here:

```
~/MCP-Toolbox/sam-toolkit-mcp/          SamToolkit, with its own .venv   (required by this harness)
~/MCP-Toolbox/langchain-harness-lab/    this repository, with its own .venv
~/LocalCodePilot/                       optional: the web interface that launches this harness
```

| Hardcoded path | Where |
|---|---|
| `~/MCP-Toolbox/sam-toolkit-mcp/.venv/bin/python`, `server.py`, `examples/` | `harness_config.py` (`SAMTOOLKIT_ROOT`), `supervisor_5b.py` (`SAMTOOLKIT_PYTHON`, `WORKSPACE_LABEL`) |
| `/Library/Frameworks/Python.framework/Versions/3.12` (the python.org installer) | `supervisor_5b.py` (`PYTHON_FRAMEWORK`): the agent and worker sandboxes may execute programs only from there, from this repository's `.venv/bin/python` and from SamToolkit's |
| this repository's `.venv/bin/python` (relative to the repository) | `supervisor_5b.py` (`PYTHON`), `run_sandboxed.sh` |
| `results/` inside this repository | supervisor, agent and evaluators (see [`results/README.md`](results/README.md)) |

## Requirements

- macOS on Apple Silicon. The harness uses `/usr/bin/sandbox-exec` (which Apple marks as deprecated),
  `libproc` and the macOS DNS resolver socket. Tested on macOS 14.4 (arm64).
- Python 3.12 **from the python.org installer**, at `/Library/Frameworks/Python.framework/Versions/3.12`.
  Tested with 3.12.6. Create both `.venv`s (this one and SamToolkit's) with that interpreter. The sandbox
  profiles allow program execution only from that location, so other Python builds (Homebrew, pyenv,
  conda) are expected to be refused. This has not been tested.
- [Ollama](https://ollama.com) running on `127.0.0.1:11434` with
  `hf.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF:Q4_K_M` (about 2.1 GB). Tested with Ollama 0.40.18.
  See [Model licence](#licences).
- SamToolkit installed at the path above (`pip install -e .` in its own `.venv`).
- At least 8 GiB free disk before every model call (`resource_guard.py`). LocalCodePilot requires 12 GiB
  free before it starts a run.

## Setup

```bash
git clone https://github.com/yige-fei/sam-toolkit-mcp.git ~/MCP-Toolbox/sam-toolkit-mcp
# set up SamToolkit first: see its README
git clone https://github.com/yige-fei/langchain-harness-lab.git ~/MCP-Toolbox/langchain-harness-lab
cd ~/MCP-Toolbox/langchain-harness-lab
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --require-hashes -r search-requirements.lock.txt
ollama pull hf.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF:Q4_K_M
```

- `requirements.lock.txt` lists the exact versions of 88 packages, without hashes.
- `search-requirements.lock.txt` hash-pins `ddgs`, `lxml` and `primp`. These are used only by the
  historical Phase 5A DuckDuckGo probe and its tests.
- The development environment matches both files exactly.
- These steps have **not** been re-run on a clean machine.

## Verify (offline, no model, no network)

```bash
.venv/bin/python -B -m unittest discover -s tests
```

Result on 2026-11-26: **267 tests OK, 4 skipped**. The four skipped tests only run inside the all-deny
sandbox, by design. The tests block sockets and use fake models and tools. Some of them start real
sandboxes and child processes. `test_isolation_5c` writes an `isolation_check_*.json` report into
`results/`, which is ignored by Git.

Read-only capability report (no model call, tool call or search):

```bash
.venv/bin/python -I supervisor_5b.py --status
```

## Running

```bash
.venv/bin/python -I supervisor_5b.py "What functions are in demo.py?" --approval tty
.venv/bin/python -I supervisor_5b.py "Search Wikipedia for Python programming language." --approval tty
```

The supervisor prints an approval card. Nothing runs until you type the card's code. LocalCodePilot starts the
supervisor with `--approval pipe --control-fd N --events-fd M` instead; see its
[architecture document](https://github.com/yige-fei/Local-Code-Pilot/blob/main/docs/ARCHITECTURE.md).

## How it works

```
supervisor_5b.py   no sandbox; standard library only; owns approvals; never opens a network connection
  |- agent:  sandbox-exec (network: localhost:11434 only; writes: results/agent_runs_5b.jsonl only)
  |            agent_5b.py -> LangGraph graph (agent_graph_5b.py)
  |            '- SamToolkit server.py over MCP stdio (SamToolkit's own .venv; inherits the same sandbox)
  '- worker: sandbox-exec (network: HTTPS 443 + system DNS socket only; no file writes)
               search_lab/wiki_worker.py -> Wikipedia MediaWiki Action API; started once per approved search
```

1. **Decide.** Qwen returns one structured decision: `inspect_python`, `inspect_csv`, `wiki_search`,
   `answer` or `ask`.
2. **Validate.** A deterministic validator (`decision_validator_v3.py`, which builds on v2 and v1) rejects
   absolute, home and `..` paths, the wrong file types, multiple files, and searches the user did not
   explicitly ask for. It approves a proposal for *human review* only. It never executes or repairs
   anything.
3. **Clarify.** When nothing should run, `clarification.py` (Phase 6D) replaces Qwen's question with a fixed
   explanation, but only when the request itself shows the problem. This layer changes wording only:
   Qwen's direct answers are never rewritten, and it can never create a card or a call.
4. **Approve.** The approval code is the first 8 hex digits of a SHA-256 digest. The digest covers the
   thread ID, a random proposal ID, the tool, its arguments, the workspace and (for searches) the fixed
   search specification. The supervisor re-checks every card itself and accepts one decision, once.
5. **Execute once.** A file approval never authorises a search. Search results are re-validated. By
   default the evidence answer quotes each snippet with its own source and adds no generated claims
   (`grounding.py`).

## Evaluation evidence (Phase 6D)

`evaluate_6d.py` ran on a fresh 48-case held-out set, written before any Phase 6D model run (hashes in
`results/phase6d_freeze.txt`). It used temperature 0 and a fixed seed. Every run stopped at the first
approval card or final text, so **no tool, MCP call or search was executed**. Reviewed totals from
[`results/eval_6d_reviewed_summary.json`](results/eval_6d_reviewed_summary.json), active decision prompt:

| Measure | Result |
|---|---|
| Qwen's raw decision correct | 30 / 48 |
| Correct first stop, strict scoring | 42 / 48 |
| Correct first stop after manual review of Qwen's refusal texts | 45 / 48 |
| False approvals (a card that should not exist) | **0** |
| Wrong tool proposals by Qwen blocked by the validator | 18 |

- 3 of 110 model calls failed with a client-side connection error before reaching Ollama. Only those 3
  were re-run, with the frozen code.
- A candidate prompt was also measured. The active decision prompt was **not** changed.
- These are single-machine results for one small model. They say nothing about model quality in general,
  and answers are not fact-checked.

## Security model and its limits

- **Network.** The agent can reach only Ollama on localhost. Only the search worker can reach the internet,
  and only after an approval. The supervisor, which runs unsandboxed as you, uses the standard library
  only and opens no network connections.
- **Environment.** `harness_config.py` removes `OLLAMA_API_KEY`, `OLLAMA_HOST` and the LangSmith/LangChain
  keys and endpoints, and turns tracing off. Child processes get a minimal environment.
- **Files.** SamToolkit reads only `.py`/`.csv` files under its `examples/` workspace. Its path checks are
  ordinary Python checks, not an operating-system sandbox, and they are not race-free.
- **Untrusted text.** Wikipedia snippets and all model output are untrusted. Model text is labelled as
  unverified.
- **Local logs.** `results/` keeps your requests, model output and approval codes on disk. They are never
  published, but they are not encrypted.

## Limitations

- Hardcoded paths: the layout above and the python.org Python 3.12 location are required. A clone
  anywhere else does not work.
- macOS-only. `sandbox-exec` is deprecated by Apple and could change in a future macOS.
- Fresh install not verified: no clean-machine install from the lock files or `git clone` has been
  tested. What has been verified is that a copy containing only the published files passes the full
  offline test suite, using the existing environment.
- One model, one tool call per request. No streaming, no file writes, no multi-step plans.
- The earlier freeze manifests (`phase5b_freeze.txt`, `phase5c_freeze.txt`) no longer match the current
  code, because later phases changed it. `evaluate_5b.py` and `evaluate_5c.py` therefore report changed
  files.

## Files

**Runtime** (imported by `supervisor_5b.py`, `agent_5b.py`, `agent_probe_5c.py` or `search_lab/wiki_worker.py`):

| File | Purpose |
|---|---|
| `supervisor_5b.py` | Entry point and the only unsandboxed process. Approval modes `tty`, `fifo`, `deny` and `pipe` (Phase 6B: `--control-fd`, `--events-fd`, signal handling, deadline, process-tree cleanup); `--status`. |
| `agent_5b.py` | Agent process started by the supervisor (stdin is `/dev/null`). |
| `agent_graph_5b.py`, `agent_graph.py` | LangGraph workflow (Phase 5B) and the Phase 4C helpers it reuses. |
| `agent_runtime_5b.py`, `agent_runtime.py` | Decision schema and prompt, Qwen calls, the MCP executor and the search hand-off. |
| `decision_validator_v3.py`, `decision_validator_v2.py`, `decision_validator.py` | Validator layers (v3 adds `wiki_search`; v2 intent layer; v1 structural/path layer). |
| `clarification.py` | Phase 6D deterministic clarification wording. |
| `grounding.py` | Deterministic evidence answers and a lexical verifier for optional model answers. |
| `wiki_capability.py`, `broker_protocol.py` | Search spec, approval digest and source checks; bounded JSON-lines pipe protocol. |
| `harness_config.py`, `resource_guard.py` | Environment hardening and fixed settings; disk/memory stop. |
| `eval_prompts.py`, `experiment_decisions.py` | Phase 4B prompt and schema helpers that the runtime still imports. |
| `agent_probe_5c.py` | Read-only probe for `--status`. |
| `search_lab/wiki_worker.py`, `search_lab/wikipedia_search.py`, `search_lab/wiki_agent_config.json` | One approved Wikipedia request (pinned host, path and parameters; redirects refused; TLS verified; 64 KiB cap; no retries) and the operator-approved User-Agent. |

**Evaluation and research history** (not used at runtime):

| File | Phase |
|---|---|
| `check_chat.py`, `check_mcp.py`, `run_agent.py`, `run_sandboxed.sh` | 4B–4C first experiments (`./run_sandboxed.sh check_chat.py`) |
| `diagnose_template.py`, `evaluate_decisions.py`, `evaluate_validator.py`, `fresh_eval_cases.py`, `adversarial_eval_v2.py` | 4B.1–4C decision and validator evaluations |
| `search_lab/ddgs_probe.py`, `search_lab/dns_mode_check.py`, `search_lab/run_search_sandboxed.sh`, `search_lab/plan_phase5a*.json`, `search_lab/run_wiki_sandboxed.sh` | 5A search probes (DuckDuckGo, then Wikipedia); superseded by the worker |
| `search_lab/isolation_check.py` | 5B/5C demonstration of the sibling sandboxes |
| `eval_5b_cases.py`, `evaluate_5b.py`, `eval_5c_cases.py`, `evaluate_5c.py`, `eval_6d_cases.py`, `evaluate_6d.py` | 5B–6D evaluations |

**Tests:** `tests/` (offline; fakes; sockets blocked).

## Licences

- **This repository's source code** is released under the [MIT Licence](LICENSE).
- **The model is not part of this repository.** Qwen2.5-Coder-3B-Instruct is distributed by its authors
  under the **Qwen RESEARCH LICENSE AGREEMENT**, not under MIT. You download it yourself through Ollama
  and must follow its own terms. This repository contains no model weights, and its MIT licence does not
  change the model's terms.
- **Wikipedia snippets** in `results/wiki_search_20261124_015451.json` are CC BY-SA 4.0 (see
  [`results/README.md`](results/README.md)).
- Python dependencies are installed from PyPI under their own licences. None are copied into this repository.
