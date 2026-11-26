"""Phase 6D: deterministic clarification wording. TEXT ONLY: it never approves, proposes or runs anything.

Used by the 5B graph in two places, both of which already end without a tool:
  - compose(): the `clarify` node (the validator rejected Qwen's decision or needs clarification). Replaces the
    one generic sentence with a specific, safe explanation.
  - refine_ask(): the `reply` node when Qwen chose action "ask". Qwen's own question is replaced ONLY when the
    request itself shows a specific problem (a path outside the workspace, several files, an unsupported file
    type); otherwise Qwen's question is kept. Qwen's "answer" replies are never changed, so negated requests and
    files mentioned in ordinary discussion keep their direct answers.

Nothing here runs before the model or instead of the validator: a pre-model short-circuit was measured on the
existing labelled sets (evaluate_6d.py) and kept out, because the request alone cannot tell discussion from an
inspection request reliably enough. The request is read with the validator's own reference analysis
(decision_validator / _v2), so "don't open demo.py" is a negated reference here exactly as it is there.
The only user text echoed back is the file name the user wrote, clipped, quoted and passed through `sanitize`
(absolute paths become "<path>"); nothing about the filesystem is looked up or revealed.
"""
from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any, Mapping

import decision_validator_v2 as v2
import decision_validator_v3 as v3
from agent_graph import sanitize
from decision_validator import TOOL_EXTENSIONS, Reason, analyze_request, path_safety_problems

GENERIC = ("I couldn't determine a safe, specific action, so nothing was run. To inspect a file, name one "
           ".py or .csv file in the workspace. To search Wikipedia, ask explicitly, for example: "
           "\"Search Wikipedia for <topic>\".")  # == agent_graph_5b.CLARIFY_TEXT (the fallback)
SUPPORTED = frozenset(TOOL_EXTENSIONS.values())
SPREADSHEETS = frozenset({".xlsx", ".xls", ".xlsm", ".ods", ".numbers", ".tsv"})
_OUTSIDE = {Reason.TRAVERSAL, Reason.ABSOLUTE_PATH, Reason.HOME_PATH, Reason.SCHEME_OR_DRIVE}
_NOT_PLAIN = {Reason.BACKSLASH, Reason.DISALLOWED_CHARACTERS, Reason.MALFORMED_PATH, Reason.PATH_TOO_LONG}

# Validator reason -> sentence, for clarifications the request analysis below does not explain better.
REASON_TEXT = {
    "negated_reference": "You asked me not to use {ref}, so nothing was run. Ask again without the file if you "
                         "want a general answer.",
    "superseded_reference": "Your request changes which file to use, so nothing was run. Please name the one "
                            "file you want, in one sentence.",
    "conflicting_instructions": "Your request both asks for and rules out the same file, so nothing was run.",
    "retracted_or_declined": "Your request cancels or declines the operation, so nothing was run.",
    "user_declined_tools": "You asked me not to use tools, so nothing was run.",
    "advice_question": "This reads as a question about a file rather than a request to inspect it, so nothing "
                        "was run. To inspect it, ask directly, for example \"How many rows are in data.csv?\".",
    "no_inspection_intent": "Your request does not clearly ask to inspect a file, so nothing was run.",
    "unsupported_operation": "I can only read: I can't edit, delete, rename, run, download or send files.",
    "answer_instead_of_inspection": "Your request asks to inspect {ref}, but Qwen answered without inspecting "
                                    "it, so that answer was not shown. Ask again to get an approval card.",
    "extension_tool_mismatch": "Qwen chose the wrong inspection tool for {path}, so nothing was run. Please "
                               "ask again.",
    "path_not_referenced": "Qwen proposed {path}, which is not the file named in your request, so nothing was "
                           "run. Please ask again, naming the exact file.",
    "multiple_actions_requested": "I can do one thing per request: either one file inspection or one Wikipedia "
                                  "search. Please ask for them separately.",
    "answer_instead_of_search": "You asked for a Wikipedia search, but Qwen answered without searching, so that "
                                "answer was not shown. Ask again to get an approval card for the search.",
    "multiple_search_requests": "I can make only one Wikipedia search per request. Please ask for one topic.",
    "search_declined": "Your request declines the search, so nothing was run.",
    "advice_or_howto_question": "This reads as a question about searching rather than a request to search, so "
                                "nothing was run. To search, ask \"Search Wikipedia for <topic>\".",
    "reported_or_conditional_request": "The search request is conditional or quoted, so nothing was run. Ask "
                                       "directly, for example \"Search Wikipedia for <topic>\".",
    "correction_in_request": "Your request contains a correction. Please restate the search in one sentence.",
    "other_provider_requested": "Wikipedia is the only search available here (no Google, Bing or general web "
                                "search), and only when you ask for it explicitly.",
    "no_explicit_search_request": "Wikipedia is searched only when you ask explicitly, for example \"Search "
                                  "Wikipedia for <topic>\".",
    "file_reference_in_request": "A Wikipedia search cannot include file names or paths.",
    "file_reference_in_query": "A Wikipedia search cannot include file names or paths.",
    "query_not_from_request": "Qwen's search query was not the topic exactly as you wrote it, so it was refused. "
                              "Please ask again.",
    "query_altered": "Qwen's search query was not the topic exactly as you wrote it, so it was refused. Please "
                     "ask again.",
    "query_has_no_topic": "The search has no clear topic. Name it, for example \"Search Wikipedia for "
                          "photosynthesis\".",
}


def _q(text: str, limit: int = 80) -> str:
    text = text if len(text) <= limit else text[:limit - 1] + "…"
    return json.dumps(text, ensure_ascii=False)


def _file_kind(action: str | None) -> str:
    return {"inspect_csv": ".csv file", "inspect_python": ".py file"}.get(action or "", ".py or .csv file")


def request_findings(request: str) -> dict[str, Any]:
    """What the request itself shows, using the validator's reference and negation analysis."""
    analysis = analyze_request(request)
    refs = analysis.references
    intents = v2._reference_intents(request, refs)
    active = [x.ref for x in intents if not x.negated and not x.superseded]
    targets = list(dict.fromkeys(r.target for r in active))
    unsafe = [r for r in active if r.unsafe]
    first_text = {r.target: r.text for r in reversed(active)}
    return {"references": [r.text for r in refs], "active": [first_text[t] for t in targets],
            "negated": [x.ref.text for x in intents if x.negated],
            "unsafe": [r.text for r in unsafe],
            "unsupported": [first_text[t] for t in targets
                            if PurePosixPath(t).suffix.lower() not in SUPPORTED and not path_safety_problems(t)],
            "search": bool(v3.find_search_commands(request))}


def _request_rule(f: Mapping[str, Any]) -> tuple[str, str] | None:
    """A rule from the request alone (unsafe path, several files, unsupported type), or None."""
    if f["unsafe"]:
        ref = f["unsafe"][0]
        problems = set(path_safety_problems(ref))
        if problems & _OUTSIDE:
            return "outside_workspace", (f"{_q(ref)} is outside the workspace, so I can't inspect it. I can only "
                                         "inspect .py and .csv files inside the workspace folder; name one of those "
                                         "instead.")
        if Reason.HIDDEN_PATH in problems:
            return "hidden_path", (f"{_q(ref)} is a hidden file or folder (its name starts with \".\"), and hidden "
                                   "files are never inspected.")
        return "not_plain_path", (f"{_q(ref)} is not a plain relative path, so I can't inspect it. Name a .py or "
                                  ".csv file inside the workspace, for example reports/data.csv.")
    if len(f["active"]) > 1:
        names = [_q(r, 60) for r in f["active"][:4]]
        listed = ", ".join(names[:-1]) + " or " + names[-1]
        text = (f"I can inspect only one file per request, so nothing was run yet. Which should I inspect first: "
                f"{listed}? Ask about that file on its own, then ask about the next one.")
        if f["unsupported"]:
            text += f" ({', '.join(_q(r, 60) for r in f['unsupported'][:3])}: only .py and .csv files can be inspected.)"
        return "one_file_per_request", text
    if f["unsupported"]:
        ref = f["unsupported"][0]
        text = (f"{_q(ref)} is not a file type I can inspect. I can only inspect .py and .csv files in the "
                "workspace, so nothing was run.")
        if PurePosixPath(ref).suffix.lower() in SPREADSHEETS:
            text += " If you save it as a .csv file in the workspace, I can inspect that."
        return "unsupported_extension", text
    return None


def compose(request: str, decision: Mapping[str, Any] | None, validation: Mapping[str, Any] | None
            ) -> tuple[str, str]:
    """(text, rule) for the clarify node. Falls back to GENERIC; never raises."""
    try:
        decision = decision if isinstance(decision, Mapping) else {}
        validation = validation if isinstance(validation, Mapping) else {}
        action = decision.get("action") if isinstance(decision.get("action"), str) else None
        path = decision.get("path") if isinstance(decision.get("path"), str) else ""
        codes = [c for k in ("structural_reasons", "intent_concerns") for c in (validation.get(k) or [])
                 if isinstance(c, str)]
        f = request_findings(request)
        search_side = action == v3.SEARCH_TOOL or f["search"] or "multiple_actions_requested" in codes
        if not search_side:
            hit = _request_rule(f)
            if hit:
                return sanitize(hit[1], 2_000), hit[0]
            if not f["references"] and (action in TOOL_EXTENSIONS or "path_not_referenced" in codes
                                        or "missing_path" in codes):
                return (f"You didn't name a file, so nothing was run; I don't guess file names. Which "
                        f"{_file_kind(action)} in the workspace should I inspect? Please give its exact name."), \
                    "missing_file_name"
        for code in codes:
            if code in REASON_TEXT:
                ref = (f["negated"] or f["active"] or [path or "the file"])[0]
                text = REASON_TEXT[code].format(ref=_q(ref), path=_q(path or "(no file name)"))
                return sanitize(text, 2_000), code
        return GENERIC, "generic"
    except Exception:  # wording must never break the graph
        return GENERIC, "generic"


def refine_ask(request: str) -> tuple[str, str] | None:
    """For Qwen's action "ask": a specific explanation when the request itself shows why it can't be done as
    asked, else None (Qwen's own question is kept). Search requests are left to the validator."""
    try:
        f = request_findings(request)
        if f["search"]:
            return None
        hit = _request_rule(f)
        return (sanitize(hit[1], 2_000), hit[0]) if hit else None
    except Exception:
        return None
