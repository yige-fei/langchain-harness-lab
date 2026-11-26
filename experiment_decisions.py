"""Step 5: test Qwen's decisions two ways. Proposed tool calls are NEVER executed.

A. Native tool calling: ChatOllama.bind_tools(<MCP tool schemas>). Only Ollama's
   structured `message.tool_calls` counts; JSON that merely appears in the text
   is recorded separately as a "text look-alike" and never treated as a call.
B. Structured decision: ChatOllama.with_structured_output(Decision,
   method="json_schema"), which makes Ollama constrain output to the schema.
"""
import harness_config as cfg  # noqa: I001  (must be first: scrubs environment)

import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal

import jsonschema
from langchain.mcp import MCPAdapter
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field

TRIALS = 3  # overridden from the command line only when run as a script (see bottom)

# (case id, user request, expected action, expected path)
CASES = [
    ("demo", "What functions are defined in demo.py?", "inspect_python", "demo.py"),
    ("sales", "How many rows are in sales.csv?", "inspect_csv", "sales.csv"),
    ("no_tool", "What is the capital of France?", "answer_directly", ""),
    ("missing", "List the functions in missing_module.py.", "inspect_python", "missing_module.py"),
    ("unspecified", "How many rows are in my CSV file?", "ask_user", ""),
]
TOOL_EXT = {"inspect_python": ".py", "inspect_csv": ".csv"}


class Decision(BaseModel):
    """One validated next step. The harness, not the model, decides whether to act on it."""
    action: Literal["inspect_python", "inspect_csv", "answer_directly", "ask_user"]
    path: str = Field(description="Exact workspace-relative file name the user gave, or empty string.")
    message: str = Field(description="Answer or clarifying question; empty string when using a tool.")


DECISION_PROMPT = cfg.SYSTEM_PROMPT + (
    " Reply only with a JSON decision with keys action, path and message."
    " action is one of: inspect_python (list functions and classes in a .py file the user named),"
    " inspect_csv (count rows and columns in a .csv file the user named),"
    " answer_directly (a general question that needs no file),"
    " ask_user (the user did not name a specific file)."
    " path is the exact file name the user gave, or an empty string."
    " message is your answer or your question to the user, or an empty string when using a tool."
)


def path_problem(action: str, path: object) -> str | None:
    """Harness-side check of a proposed path. Returns a problem description or None."""
    if not isinstance(path, str) or not path.strip():
        return "empty or non-string path"
    p = PurePosixPath(path)
    if p.is_absolute() or path.startswith("~") or "\\" in path:
        return "not a relative POSIX path"
    if ".." in p.parts:
        return "parent traversal"
    if not re.fullmatch(r"[A-Za-z0-9_.\-/]{1,200}", path):
        return "unexpected characters"
    if p.suffix != TOOL_EXT[action]:
        return f"wrong extension for {action}"
    return None


def text_lookalike(content: str) -> str | None:
    """Detect JSON in plain text that RESEMBLES a tool call (diagnostic only, never executed)."""
    m = re.search(r'\{[^{}]*"name"\s*:\s*"(\w+)"', content or "")
    return m.group(1) if m else None


def score(expected_action: str, expected_path: str, action: str | None, path: object,
          arguments_ok: bool, message: str) -> str:
    if action != expected_action:
        return f"WRONG: {action or 'no tool call'} (expected {expected_action})"
    if action in TOOL_EXT:
        if not arguments_ok:
            return "WRONG: arguments fail MCP input schema"
        problem = path_problem(action, path)
        if problem:
            return f"WRONG: invalid path {path!r} ({problem})"
        return "CORRECT" if path == expected_path else f"WRONG: path {path!r} (expected {expected_path!r})"
    if expected_action == "answer_directly" and "paris" not in message.lower():
        return "WRONG: no-tool reply but answer missing 'Paris'"
    return "CORRECT"


async def discover_schemas() -> tuple[list[dict], dict[str, dict]]:
    """Get tool schemas from the live MCP server, then disconnect. Returns plain dicts only."""
    async with MCPAdapter(cfg.make_mcp_transport()) as adapter:
        tools = cfg.require_allowed_tools(await adapter.list_tools())
    return [convert_to_openai_tool(t) for t in tools], {t.name: t.args_schema for t in tools}


def timing(meta: dict) -> dict:
    return {"total_s": round((meta.get("total_duration") or 0) / 1e9, 2),
            "out_tokens": meta.get("eval_count"), "prompt_tokens": meta.get("prompt_eval_count")}


def run_native(llm, tool_dicts, input_schemas, log) -> list[str]:
    bound = llm.bind_tools(tool_dicts)  # schemas only: nothing here can execute a tool
    verdicts = []
    for cid, prompt, exp_action, exp_path in CASES:
        for trial in range(1, TRIALS + 1):
            msg = bound.invoke([("system", cfg.SYSTEM_PROMPT), ("user", prompt)])
            calls = msg.tool_calls
            action = calls[0]["name"] if calls else ("answer_directly" if exp_action == "answer_directly" else None)
            if not calls and exp_action == "ask_user":
                action = "ask_user"  # a reply with no call is the right shape for a clarifying question
            args = calls[0]["args"] if calls else {}
            args_ok = bool(calls) and calls[0]["name"] in input_schemas and _valid(args, input_schemas[calls[0]["name"]])
            verdict = score(exp_action, exp_path, action, args.get("path"), args_ok, msg.content)
            lookalike = text_lookalike(msg.content)
            notes = []
            if len(calls) > 1:
                notes.append(f"{len(calls)} native calls")
            if msg.invalid_tool_calls:
                notes.append(f"invalid_tool_calls={msg.invalid_tool_calls}")
            if lookalike:
                notes.append(f"TEXT LOOK-ALIKE call to {lookalike!r} (not a native call)")
                if verdict == "CORRECT":
                    verdict = "WRONG: tool-call-shaped JSON in text"
            verdicts.append(verdict)
            rec = {"experiment": "A-native", "case": cid, "trial": trial, "verdict": verdict, "notes": notes,
                   "native_tool_calls": calls, "content": msg.content, **timing(msg.response_metadata)}
            log.write(json.dumps(rec) + "\n")
            print(f"[A native] {cid:11} t{trial}: {verdict} {notes}\n   tool_calls={json.dumps(calls)}"
                  f"\n   content={msg.content!r}  {timing(msg.response_metadata)}")
    return verdicts


def run_structured(llm, input_schemas, log) -> list[str]:
    decider = llm.with_structured_output(Decision, method="json_schema", include_raw=True)
    verdicts = []
    for cid, prompt, exp_action, exp_path in CASES:
        for trial in range(1, TRIALS + 1):
            out = decider.invoke([("system", DECISION_PROMPT), ("user", prompt)])
            raw, parsed, err = out["raw"], out["parsed"], out["parsing_error"]
            if parsed is None:
                verdict = f"WRONG: output failed Decision schema ({type(err).__name__})"
            else:
                args_ok = parsed.action in input_schemas and _valid({"path": parsed.path}, input_schemas[parsed.action])
                verdict = score(exp_action, exp_path, parsed.action, parsed.path, args_ok, parsed.message)
                if parsed.action in TOOL_EXT and parsed.message:
                    verdict += " (note: non-empty message with tool action)"
            verdicts.append(verdict)
            rec = {"experiment": "B-structured", "case": cid, "trial": trial, "verdict": verdict,
                   "parsed": parsed.model_dump() if parsed else None, "parsing_error": repr(err) if err else None,
                   "raw_content": raw.content, "raw_native_tool_calls": raw.tool_calls, **timing(raw.response_metadata)}
            log.write(json.dumps(rec) + "\n")
            print(f"[B struct] {cid:11} t{trial}: {verdict}\n   raw={raw.content!r}  {timing(raw.response_metadata)}")
    return verdicts


def _valid(instance: dict, schema: dict) -> bool:
    try:
        jsonschema.validate(instance, schema)
        return True
    except jsonschema.ValidationError:
        return False


def main() -> None:
    tool_dicts, input_schemas = asyncio.run(discover_schemas())
    llm = cfg.make_llm()
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    out = cfg.RESULTS_DIR / f"decisions_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    start = time.perf_counter()
    with out.open("w") as log:
        a = run_native(llm, tool_dicts, input_schemas, log)
        b = run_structured(llm, input_schemas, log)
    print(f"\nA native tool calling : {sum(v == 'CORRECT' for v in a)}/{len(a)} correct")
    print(f"B structured decisions: {sum(v.startswith('CORRECT') for v in b)}/{len(b)} correct")
    print(f"model calls: {len(a) + len(b)}  wall time: {time.perf_counter() - start:.1f}s  log: {out}")


if __name__ == "__main__":
    TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    main()
