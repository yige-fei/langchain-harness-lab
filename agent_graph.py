"""Phase 4C: minimal LangGraph workflow with validated decisions and human approval.

    START -> decide -> validate -+-> prepare_proposal -> await_approval -+-> execute -> answer -> END
                                 |                                       +-> denied ----------> END
                                 +-> reply (validated answer / question) -------------------> END
                                 +-> clarify (rejected or uncertain: fixed text) -----------> END
                                 +-> fail (unexpected error: fixed text) -------------------> END

The model, the tool executor and the answer writer are injected, so tests can use
fakes. This module never imports a model client or MCP; it cannot reach the network.

Safety properties enforced here (not by the model):
- a tool runs only after an explicit human approval bound to the exact proposal
  (thread, proposal id, tool, arguments, workspace) via a SHA-256 digest;
- at most one tool call per request, no retries, no tool after the answer step;
- anything unexpected fails closed with a fixed message.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import operator
import re
import uuid
from typing import Annotated, Any, Awaitable, Callable, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import decision_validator_v2 as validator

ALLOWED_TOOLS = frozenset({"inspect_python", "inspect_csv"})
MAX_TOOL_CALLS = 1
TOOL_TIMEOUT_S = 30
ANSWER_MAX_CHARS = 2_000
RECURSION_LIMIT = 12

CLARIFY_TEXT = ("I couldn't determine a safe, specific action, so nothing was run. Please name one .py or .csv "
                "file in the workspace and say what you want to know about it.")
DENIED_TEXT = "Approval was not given, so no tool was run."
FAILURE_TEXT = "Something went wrong before any tool could run safely, so nothing was run."
TOOL_FAILED_TEXT = "The tool could not complete the request."
FALLBACK_PREFIX = "[Fallback answer, not written by the model]"

DecideFn = Callable[[str], Awaitable[dict]]
ExecuteFn = Callable[[str, dict], Awaitable[dict]]  # -> {"status": "success"|"error", "text": str, "structured": dict|None}
AnswerFn = Callable[[str, dict], Awaitable[str]]


class AgentState(TypedDict, total=False):
    request: str
    decision: dict | None
    validation: dict | None
    proposal: dict | None
    approval: dict | None
    tool_calls_made: int
    tool_result: dict | None
    final: str
    route: str
    failure_hint: str | None
    events: Annotated[list[str], operator.add]


# --- helpers -------------------------------------------------------------------
_ABS_PATH = re.compile(r"(?<![\w.])(?:~?/[^\s\"'<>]+|[A-Za-z]:\\[^\s\"'<>]+)")
_TOOL_CALL_TEXT = re.compile(r"<tool_call>|\"name\"\s*:\s*\"inspect_|\binspect_(?:python|csv)\s*\(", re.I)


_SAFE_DETAIL = re.compile(r"[A-Za-z0-9_:=<>.\- ]{1,120}")  # no "/", "\\", "~" or quotes: cannot hold a path


def exception_audit(exc: BaseException) -> tuple[list[str], str | None]:
    """Per-request audit events and a user hint for an exception that OPTS IN by defining
    `audit_detail` / `user_hint` (e.g. ResourceLimitReached). Other exceptions contribute
    only their type name elsewhere; their messages are never recorded."""
    detail, hint = getattr(exc, "audit_detail", None), getattr(exc, "user_hint", None)
    events = [f"guard:{detail}"] if isinstance(detail, str) and _SAFE_DETAIL.fullmatch(detail) else []
    hint = hint if isinstance(hint, str) and _SAFE_DETAIL.fullmatch(hint) else None
    return events, hint


def sanitize(text: object, limit: int = 1_000) -> str:
    """Remove absolute paths and cap length before anything reaches the user."""
    return _ABS_PATH.sub("<path>", str(text))[:limit]


def proposal_digest(thread_id: str, proposal: dict) -> str:
    core = {k: proposal[k] for k in ("proposal_id", "tool", "arguments", "workspace")}
    core["thread_id"] = thread_id
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()


def approval_code(digest: str) -> str:
    return digest[:8].upper()


def approval_card(proposal: dict) -> dict:
    """Exactly what the human is asked to approve."""
    return {"type": "tool_approval_request", "tool": proposal["tool"], "arguments": proposal["arguments"],
            "workspace": proposal["workspace"], "user_reference": proposal["user_reference"],
            "request": proposal["request"], "proposal_id": proposal["proposal_id"],
            "digest": proposal["digest"], "approval_code": approval_code(proposal["digest"])}


def human_response(card: dict, approve: bool, via: str) -> dict:
    """Build the resume value. Callers must only pass approve=True after a human said yes."""
    return {"decision": "approve" if approve else "deny", "proposal_id": card["proposal_id"],
            "digest": card["digest"], "via": via}


def deterministic_summary(tool: str, structured: dict | None) -> str:
    if not isinstance(structured, dict):
        return "no structured result"
    if tool == "inspect_python":
        funcs = ", ".join(f"{f.get('name')} (line {f.get('line')})" for f in structured.get("functions", [])) or "none"
        classes = ", ".join(f"{c.get('name')} (methods: {', '.join(c.get('methods', [])) or 'none'})"
                            for c in structured.get("classes", [])) or "none"
        return f"{structured.get('file')}: functions: {funcs}; classes: {classes}"
    if tool == "inspect_csv":
        return (f"{structured.get('file')}: {structured.get('rows')} rows; columns: "
                f"{', '.join(structured.get('columns', []))}; missing values: {structured.get('missing_values')}")
    return "unsupported tool"


def execution_allowed(state: dict, thread_id: str) -> bool:
    """Final gate before the single tool call. Re-checks everything instead of trusting earlier nodes."""
    try:
        proposal, approval, v = state["proposal"], state["approval"], state["validation"]
        recheck = validator.validate_decision(state["request"], state["decision"])
        return all([
            isinstance(approval, dict) and approval.get("granted") is True,
            approval["digest"] == proposal["digest"] == proposal_digest(thread_id, proposal),
            approval["proposal_id"] == proposal["proposal_id"],
            proposal["tool"] in ALLOWED_TOOLS and proposal["tool"] == v["action"],
            set(proposal["arguments"]) == {"path"} and proposal["arguments"]["path"] == v["tool_path"],
            recheck.ready_for_human_review and recheck.tool_path == v["tool_path"] and recheck.action == v["action"],
            state.get("tool_calls_made", 0) < MAX_TOOL_CALLS,
        ])
    except Exception:
        return False


def _thread_id(config: RunnableConfig) -> str:
    tid = (config or {}).get("configurable", {}).get("thread_id")
    if not tid:
        raise ValueError("thread_id is required")
    return str(tid)


# --- graph -----------------------------------------------------------------------
def build_graph(decide_fn: DecideFn, execute_fn: ExecuteFn, answer_fn: AnswerFn, workspace_label: str,
                checkpointer=None):
    async def decide(state: AgentState) -> dict:
        try:
            decision = await decide_fn(state["request"])
            if not isinstance(decision, dict):
                raise TypeError("decision must be a dict")
            return {"decision": decision, "events": ["decided"]}
        except GraphBubbleUp:
            raise
        except Exception as exc:  # fail closed
            guard_events, hint = exception_audit(exc)
            return {"decision": None, "route": "fail", "failure_hint": hint,
                    "events": [f"decide_failed:{type(exc).__name__}", *guard_events]}

    async def validate(state: AgentState) -> dict:
        if state.get("route") == "fail":
            return {}
        try:
            r = validator.validate_decision(state["request"], state["decision"])
            v = {"outcome": r.outcome.value, "action": r.action, "tool_path": r.tool_path,
                 "user_reference": r.user_reference,
                 "structural_reasons": [x.value for x in r.structural_reasons],
                 "intent_concerns": [x.value for x in r.intent_concerns]}
            if r.ready_for_human_review:
                route = "propose"
            elif r.outcome is validator.Outcome.APPROVED_FOR_REVIEW and r.action in ("answer", "ask"):
                route = "reply"
            else:
                route = "clarify"
            return {"validation": v, "route": route, "events": [f"validated:{r.outcome.value}"]}
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return {"validation": None, "route": "fail", "events": [f"validate_failed:{type(exc).__name__}"]}

    async def prepare_proposal(state: AgentState, config: RunnableConfig) -> dict:
        # Runs once, BEFORE the interrupt node, so the proposal id is stable across resume.
        v = state["validation"]
        proposal = {"proposal_id": uuid.uuid4().hex, "tool": v["action"], "arguments": {"path": v["tool_path"]},
                    "workspace": workspace_label, "user_reference": v["user_reference"], "request": state["request"]}
        proposal["digest"] = proposal_digest(_thread_id(config), proposal)
        return {"proposal": proposal, "events": ["proposal_prepared"]}

    async def await_approval(state: AgentState, config: RunnableConfig) -> dict:
        proposal = state["proposal"]
        response = interrupt(approval_card(proposal))  # pauses here; re-runs from the top on resume
        expected = proposal_digest(_thread_id(config), proposal)
        granted = (isinstance(response, dict) and response.get("decision") == "approve"
                   and response.get("proposal_id") == proposal["proposal_id"]
                   and response.get("digest") == proposal["digest"] == expected
                   and isinstance(response.get("via"), str) and bool(response["via"]))
        approval = {"granted": granted, "proposal_id": proposal["proposal_id"], "digest": proposal["digest"],
                    "via": response.get("via") if isinstance(response, dict) else None}
        return {"approval": approval, "route": "execute" if granted else "denied",
                "events": ["approval_granted" if granted else "approval_denied"]}

    async def execute(state: AgentState, config: RunnableConfig) -> dict:
        try:
            allowed = execution_allowed(state, _thread_id(config))
        except Exception:
            allowed = False
        if not allowed:
            return {"route": "blocked", "final": FAILURE_TEXT, "events": ["execution_blocked_by_guard"]}
        proposal = state["proposal"]
        try:
            result = await asyncio.wait_for(execute_fn(proposal["tool"], dict(proposal["arguments"])), TOOL_TIMEOUT_S)
            status = result.get("status") if isinstance(result, dict) else None
            if status not in ("success", "error"):
                raise ValueError("bad tool result")
            tool_result = {"status": status, "text": sanitize(result.get("text", "")),
                           "structured": result.get("structured") if status == "success" else None}
        except GraphBubbleUp:
            raise
        except Exception as exc:  # no retry
            tool_result = {"status": "error", "text": TOOL_FAILED_TEXT, "structured": None,
                           "exception": type(exc).__name__}
        return {"tool_calls_made": state.get("tool_calls_made", 0) + 1, "tool_result": tool_result,
                "route": "answer", "events": [f"tool_executed:{tool_result['status']}"]}

    async def answer(state: AgentState) -> dict:
        tool, result = state["proposal"]["tool"], state["tool_result"]
        if result["status"] != "success":
            return {"final": f"The tool reported an error: {sanitize(result['text'])}", "events": ["answered:error"]}
        summary = deterministic_summary(tool, result["structured"])
        source = f"[Source: {tool}({json.dumps(state['proposal']['arguments'])}) -> {summary}]"
        try:
            # The answer step gets DATA only; it has no tools and its text never triggers one.
            text = await asyncio.wait_for(answer_fn(state["request"], result), TOOL_TIMEOUT_S * 4)
            text = sanitize(text, ANSWER_MAX_CHARS).strip() if isinstance(text, str) else ""
            if text and not _TOOL_CALL_TEXT.search(text):
                return {"final": f"{text}\n\n{source}", "events": ["answered:model"]}
            reason, events = "the model's reply was empty or looked like a tool call, so it was discarded", \
                ["answered:fallback"]
        except GraphBubbleUp:
            raise
        except Exception as exc:
            guard_events, hint = exception_audit(exc)
            reason = (f"the model answer was skipped because a local resource limit was reached ({hint})"
                      if hint else "the model answer was unavailable because of an error")
            events = [f"answered:fallback:{type(exc).__name__}", *guard_events]
        label = f"{FALLBACK_PREFIX} {reason[0].upper()}{reason[1:]}. The verified tool result is shown below."
        return {"final": f"{label}\n\n{source}", "events": events}

    async def reply(state: AgentState) -> dict:
        return {"final": sanitize(state["decision"]["message"], ANSWER_MAX_CHARS), "events": ["replied"]}

    async def clarify(state: AgentState) -> dict:
        return {"final": CLARIFY_TEXT, "events": ["clarified"]}

    async def denied(state: AgentState) -> dict:
        return {"final": DENIED_TEXT, "events": ["denied"]}

    async def fail(state: AgentState) -> dict:
        hint = state.get("failure_hint")
        final = f"{FAILURE_TEXT} (Stopped because a local resource limit was reached: {hint}.)" if hint else FAILURE_TEXT
        return {"final": final, "events": ["failed_closed"]}

    g = StateGraph(AgentState)
    for name, fn in [("decide", decide), ("validate", validate), ("prepare_proposal", prepare_proposal),
                     ("await_approval", await_approval), ("execute", execute), ("answer", answer),
                     ("reply", reply), ("clarify", clarify), ("denied", denied), ("fail", fail)]:
        g.add_node(name, fn)
    g.add_edge(START, "decide")
    g.add_edge("decide", "validate")
    g.add_conditional_edges("validate", lambda s: s.get("route", "fail"),
                            {"propose": "prepare_proposal", "reply": "reply", "clarify": "clarify", "fail": "fail"})
    g.add_edge("prepare_proposal", "await_approval")
    g.add_conditional_edges("await_approval", lambda s: s["route"], {"execute": "execute", "denied": "denied"})
    g.add_conditional_edges("execute", lambda s: s["route"], {"answer": "answer", "blocked": END})
    for terminal in ("answer", "reply", "clarify", "denied", "fail"):
        g.add_edge(terminal, END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


# --- driver ------------------------------------------------------------------------
def config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


async def _status(graph, thread_id: str) -> dict:
    snap = await graph.aget_state(config_for(thread_id))
    if snap.interrupts:
        return {"status": "awaiting_approval", "card": snap.interrupts[0].value, "thread_id": thread_id}
    v = snap.values
    return {"status": "done", "final": v.get("final", FAILURE_TEXT), "thread_id": thread_id,
            "tool_calls_made": v.get("tool_calls_made", 0), "events": v.get("events", []),
            "validation": v.get("validation"), "decision": v.get("decision")}


def _failed(thread_id: str, exc: Exception) -> dict:
    return {"status": "done", "final": FAILURE_TEXT, "thread_id": thread_id, "error": type(exc).__name__}


async def start(graph, request: str, thread_id: str) -> dict:
    snap = await graph.aget_state(config_for(thread_id))
    if snap.values:
        raise ValueError("thread_id already used; each request needs a new thread")
    try:
        await graph.ainvoke({"request": request, "tool_calls_made": 0, "events": []}, config_for(thread_id))
        return await _status(graph, thread_id)
    except Exception as exc:  # fail closed
        return _failed(thread_id, exc)


async def resume(graph, thread_id: str, response: object) -> dict:
    """Resume a paused thread. Refuses threads that are not waiting for approval.

    Anything other than a dict WITH a top-level "decision" key is converted into an explicit denial before
    it reaches LangGraph. (Phase 5C fix: LangGraph reads a dict whose keys all look like interrupt ids -
    including the empty dict - as a per-interrupt resume map, so `{}` used to leave the thread paused and
    an id-keyed map was a second, undocumented resume format. human_response() always sets "decision",
    so every legitimate caller is unaffected.)
    """
    snap = await graph.aget_state(config_for(thread_id))
    if not snap.interrupts:
        raise ValueError("this thread is not waiting for approval")
    if not isinstance(response, dict) or "decision" not in response:
        response = {"decision": "deny", "via": "invalid_response"}
    try:
        await graph.ainvoke(Command(resume=response), config_for(thread_id))
        return await _status(graph, thread_id)
    except Exception as exc:  # fail closed
        return _failed(thread_id, exc)
