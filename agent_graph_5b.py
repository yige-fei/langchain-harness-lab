"""Phase 5B: LangGraph workflow with the two SamToolkit file tools plus ONE `wiki_search` action.

    START -> decide -> validate -+-> prepare_proposal -> await_approval -+-> execute -> answer -> END
                                 |                                       +-> denied ----------> END
                                 +-> reply / clarify / fail (fixed text) -------------------> END

Same shape and guarantees as the Phase 4C graph (agent_graph.py, unchanged), with these additions:
- decisions are validated by decision_validator_v3 (query field; file contract delegated to v2);
- a wiki_search proposal binds the exact query AND the full search spec (provider, host, endpoint,
  parameters, User-Agent, max results, timeout, retries, what is transmitted) into the approval digest;
- the search itself is NOT performed here: execute_fn hands it to the supervisor, which runs the
  separately sandboxed worker (see supervisor_5b.py). This module never imports a network client;
- search results are re-validated, given to the answer step as untrusted data, and the final text
  always carries a locally built source list. The answer step can never start another tool.

The model, the executor and the answer writer are injected, so tests use fakes.
"""
from __future__ import annotations

import asyncio
import json
import operator
import re
import uuid
from typing import Annotated, Awaitable, Callable, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

import agent_graph as ag4                    # Phase 4C helpers and driver, reused unchanged
import clarification                        # Phase 6D: deterministic wording only (never runs anything)
import decision_validator_v3 as validator
import grounding
import wiki_capability as wc
from agent_graph import (DENIED_TEXT, FAILURE_TEXT, FALLBACK_PREFIX, TOOL_FAILED_TEXT,  # noqa: F401 (re-export)
                         config_for, deterministic_summary, exception_audit, human_response, sanitize, start)


async def resume(graph, thread_id: str, response: object) -> dict:
    """Phase 4C resume, plus: anything that is not a dict WITH a "decision" key becomes an explicit denial.

    (LangGraph reads a dict whose keys all look like interrupt ids - including the empty dict - as a
    per-interrupt resume map; `{}` would leave the thread paused instead of denying it.)
    """
    if not isinstance(response, dict) or "decision" not in response:
        response = {"decision": "deny", "via": "invalid_response"}
    return await ag4.resume(graph, thread_id, response)

FILE_TOOLS = frozenset({"inspect_python", "inspect_csv"})
SEARCH_TOOL = wc.SEARCH_TOOL
ALLOWED_TOOLS = FILE_TOOLS | {SEARCH_TOOL}
MAX_TOOL_CALLS = 1
TOOL_TIMEOUT_S = 30           # file tools (as in 4C); also bounds the hand-off to the search worker
ANSWER_TIMEOUT_S = 120
ANSWER_MAX_CHARS = 2_000
RECURSION_LIMIT = ag4.RECURSION_LIMIT

CLARIFY_TEXT = ("I couldn't determine a safe, specific action, so nothing was run. To inspect a file, name one "
                ".py or .csv file in the workspace. To search Wikipedia, ask explicitly, for example: "
                "\"Search Wikipedia for <topic>\".")

DecideFn = Callable[[str], Awaitable[dict]]
# execute_fn(tool, arguments, binding) -> file tools: {"status","text","structured"}; wiki_search: worker record
ExecuteFn = Callable[[str, dict, dict], Awaitable[dict]]
AnswerFn = Callable[[str, dict], Awaitable[str]]

_TOOL_CALL_TEXT = re.compile(ag4._TOOL_CALL_TEXT.pattern + r"|\bwiki_search\b|\"action\"\s*:", re.I)
proposal_digest = wc.proposal_digest
approval_code = wc.approval_code


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


def approval_card(proposal: dict, thread_id: str) -> dict:
    """Exactly what the human is asked to approve (shown by the supervisor, outside the agent)."""
    card = {"type": "tool_approval_request", "tool": proposal["tool"], "arguments": proposal["arguments"],
            "workspace": proposal["workspace"], "user_reference": proposal["user_reference"],
            "request": proposal["request"], "proposal_id": proposal["proposal_id"], "thread_id": thread_id,
            "digest": proposal["digest"], "approval_code": approval_code(proposal["digest"])}
    if "search_spec" in proposal:
        card["search_spec"] = proposal["search_spec"]
    return card


def execution_allowed(state: dict, thread_id: str, search_spec: dict) -> bool:
    """Final gate before the single tool call. Re-checks everything instead of trusting earlier nodes."""
    try:
        proposal, approval, v = state["proposal"], state["approval"], state["validation"]
        tool, args = proposal["tool"], proposal["arguments"]
        recheck = validator.validate_decision(state["request"], state["decision"])
        common = [
            isinstance(approval, dict) and approval.get("granted") is True,
            approval["digest"] == proposal["digest"] == proposal_digest(thread_id, proposal),
            approval["proposal_id"] == proposal["proposal_id"],
            tool in ALLOWED_TOOLS and tool == v["action"] == recheck.action,
            recheck.ready_for_human_review,
            state.get("tool_calls_made", 0) < MAX_TOOL_CALLS,
        ]
        if tool == SEARCH_TOOL:
            specific = [set(args) == {"query"}, args["query"] == v["query"] == recheck.query,
                        proposal.get("search_spec") == search_spec,
                        wc.wiki.query_problem(args["query"]) is None]
        else:
            specific = [set(args) == {"path"}, args["path"] == v["tool_path"] == recheck.tool_path,
                        "search_spec" not in proposal]
        return all(common) and all(specific)
    except Exception:
        return False


def _guarded_text(text: object, allowed_urls: set[str]) -> str:
    """Sanitize model text without mangling the (already verified) result URLs."""
    text = text if isinstance(text, str) else ""
    keep = sorted(allowed_urls, key=len, reverse=True)
    for i, url in enumerate(keep):
        text = text.replace(url, f"\x00{i}\x00")
    text = sanitize(text, ANSWER_MAX_CHARS * 2)
    for i, url in enumerate(keep):
        text = text.replace(f"\x00{i}\x00", url)
    return text[:ANSWER_MAX_CHARS].strip()


def build_graph(decide_fn: DecideFn, execute_fn: ExecuteFn, answer_fn: AnswerFn, workspace_label: str,
                search_spec: dict, checkpointer=None, answer_mode: str = "evidence"):
    """answer_mode for Wikipedia results: "evidence" (default; deterministic quotation, no model) or
    "model" (Qwen writes the answer; it is shown only if grounding.verify_answer accepts every sentence)."""
    if answer_mode not in ("evidence", "model"):
        raise ValueError("answer_mode must be 'evidence' or 'model'")
    if search_spec != wc.search_spec_for(search_spec.get("user_agent", "")):
        raise ValueError("search_spec must be the fixed Wikipedia spec")

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
            v = {"outcome": r.outcome.value, "action": r.action, "tool_path": r.tool_path, "query": r.query,
                 "user_reference": r.user_reference, "structural_reasons": list(r.structural_reasons),
                 "intent_concerns": list(r.intent_concerns)}
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
        proposal = {"proposal_id": uuid.uuid4().hex, "tool": v["action"], "workspace": workspace_label,
                    "user_reference": v["user_reference"], "request": state["request"]}
        if v["action"] == SEARCH_TOOL:
            proposal["arguments"] = {"query": v["query"]}
            proposal["search_spec"] = dict(search_spec)
        else:
            proposal["arguments"] = {"path": v["tool_path"]}
        proposal["digest"] = proposal_digest(ag4._thread_id(config), proposal)
        return {"proposal": proposal, "events": ["proposal_prepared"]}

    async def await_approval(state: AgentState, config: RunnableConfig) -> dict:
        proposal, thread_id = state["proposal"], ag4._thread_id(config)
        response = interrupt(approval_card(proposal, thread_id))  # pauses; re-runs from the top on resume
        expected = proposal_digest(thread_id, proposal)
        granted = (isinstance(response, dict) and response.get("decision") == "approve"
                   and response.get("proposal_id") == proposal["proposal_id"]
                   and response.get("digest") == proposal["digest"] == expected
                   and isinstance(response.get("via"), str) and bool(response["via"]))
        approval = {"granted": granted, "proposal_id": proposal["proposal_id"], "digest": proposal["digest"],
                    "via": response.get("via") if isinstance(response, dict) else None}
        return {"approval": approval, "route": "execute" if granted else "denied",
                "events": ["approval_granted" if granted else "approval_denied"]}

    async def execute(state: AgentState, config: RunnableConfig) -> dict:
        thread_id = ag4._thread_id(config)
        if not execution_allowed(state, thread_id, search_spec):
            return {"route": "blocked", "final": FAILURE_TEXT, "events": ["execution_blocked_by_guard"]}
        proposal = state["proposal"]
        tool, args = proposal["tool"], dict(proposal["arguments"])
        binding = {"thread_id": thread_id, "proposal_id": proposal["proposal_id"], "digest": proposal["digest"]}
        try:
            raw = await asyncio.wait_for(execute_fn(tool, args, binding), TOOL_TIMEOUT_S)
            if tool == SEARCH_TOOL:
                tool_result = wc.normalize_search_record(raw, args["query"])
            else:
                status = raw.get("status") if isinstance(raw, dict) else None
                if status not in ("success", "error"):
                    raise ValueError("bad tool result")
                tool_result = {"status": status, "tool": tool, "text": sanitize(raw.get("text", "")),
                               "structured": raw.get("structured") if status == "success" else None}
        except GraphBubbleUp:
            raise
        except Exception as exc:  # no retry
            tool_result = {"status": "error", "tool": tool, "structured": None, "exception": type(exc).__name__,
                           "text": (wc.error_text({"status": "worker_timeout"}) if isinstance(exc, TimeoutError)
                                    and tool == SEARCH_TOOL else TOOL_FAILED_TEXT)}
        return {"tool_calls_made": state.get("tool_calls_made", 0) + 1, "tool_result": tool_result,
                "route": "answer", "events": [f"tool_executed:{tool_result['status']}"]}

    async def answer(state: AgentState) -> dict:
        tool, result = state["proposal"]["tool"], state["tool_result"]
        if tool == SEARCH_TOOL:
            return await _answer_search(state["request"], state["proposal"]["arguments"]["query"], result)
        if result["status"] != "success":
            return {"final": f"The tool reported an error: {sanitize(result['text'])}", "events": ["answered:error"]}
        summary = deterministic_summary(tool, result["structured"])
        source = f"[Source: {tool}({json.dumps(state['proposal']['arguments'])}) -> {summary}]"
        try:
            text = await asyncio.wait_for(answer_fn(state["request"], result), ANSWER_TIMEOUT_S)
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

    async def _answer_search(request: str, query: str, result: dict) -> dict:
        if result["status"] != "success":
            return {"final": f"The Wikipedia search did not complete. {result['text']}",
                    "events": [f"answered:search_error:{result.get('search_status')}"]}
        results = result["structured"]["results"]
        if not results:
            return {"final": f'Wikipedia returned no matching articles for "{query}". No other search was made.',
                    "events": ["answered:no_results"]}
        evidence, _ = grounding.evidence_answer(request, query, results)
        evidence = f"{evidence}\n\n({wc.LICENSE_NOTE})"
        if answer_mode == "evidence":  # default: no model, no generated claims
            return {"final": evidence, "events": ["answered:evidence"]}
        allowed = {r["url"] for r in results}
        try:
            # DATA only: the answer step has no tools and its text never triggers one.
            raw = await asyncio.wait_for(answer_fn(request, result), ANSWER_TIMEOUT_S)
            raw = raw if isinstance(raw, str) else ""
            foreign = wc.foreign_urls(raw, allowed)  # on the RAW text: sanitizing could hide a URL
            text = _guarded_text(raw, allowed)
            accepted, checks, problems = grounding.verify_answer(text, results) if text else (False, [], ["empty"])
            if text and not _TOOL_CALL_TEXT.search(text) and not foreign and accepted:
                return {"final": f"{text}\n\n{wc.sources_block(query, results)}",
                        "events": ["answered:model:verified"]}
            if foreign:
                reason, events = ("the model's reply cited a source that was not in the search results, so it was "
                                  "discarded"), ["answered:fallback:foreign_source"]
            elif not text or _TOOL_CALL_TEXT.search(text):
                reason, events = ("the model's reply was empty or looked like a tool call, so it was discarded",
                                  ["answered:fallback"])
            else:
                reason = ("the model's reply was withheld because the deterministic check could not match it to the "
                          f"snippets ({'; '.join(problems)})")
                events = ["answered:fallback:unsupported_claims"]
        except GraphBubbleUp:
            raise
        except Exception as exc:
            guard_events, hint = exception_audit(exc)
            reason = (f"the model answer was skipped because a local resource limit was reached ({hint})"
                      if hint else "the model answer was unavailable because of an error")
            events = [f"answered:fallback:{type(exc).__name__}", *guard_events]
        label = f"{FALLBACK_PREFIX} {reason[0].upper()}{reason[1:]}. The snippets are quoted below instead."
        return {"final": f"{label}\n\n{evidence}", "events": events}

    async def reply(state: AgentState) -> dict:
        # Phase 6D: Qwen's "ask" is replaced by a fixed explanation only when the request itself shows the
        # problem (outside-workspace path, several files, unsupported type). "answer" text is never changed.
        if state["decision"].get("action") == "ask":
            refined = clarification.refine_ask(state["request"])
            if refined:
                return {"final": refined[0], "events": ["replied", f"ask_refined:{refined[1]}"]}
        return {"final": sanitize(state["decision"]["message"], ANSWER_MAX_CHARS), "events": ["replied"]}

    async def clarify(state: AgentState) -> dict:
        # Phase 6D: a specific, fixed explanation instead of one generic sentence (CLARIFY_TEXT is the fallback).
        text, rule = clarification.compose(state["request"], state.get("decision"), state.get("validation"))
        return {"final": text, "events": ["clarified", f"clarification:{rule}"]}

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
