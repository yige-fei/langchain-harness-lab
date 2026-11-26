"""Phase 4C decision validator: two separate layers.

Layer 1 - STRUCTURE AND PATH (hard rules, reused unchanged from the frozen Phase 4B.2
validator): decision shape, allowed actions, contradictions, path syntax, extensions,
exact match against a complete file reference the user wrote, unsafe references,
injection markers. Any failure -> REJECTED.

Layer 2 - INTENT ASSESSMENT (heuristic): negation tied to the file, retractions,
advice questions, corrections, conflicts, multiple targets, unsupported operations,
missing inspection wording. Any concern -> NEEDS_CLARIFICATION.

Keyword matching does NOT prove that the user authorized anything. The best outcome
here is APPROVED_FOR_REVIEW: the proposal may be shown to a human, who decides.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Mapping

from decision_validator import (  # Layer 1 building blocks, unchanged from Phase 4B.2
    MAX_USER_CHARS, TOOL_EXTENSIONS, FileReference, ModelDecision, Reason, analyze_request,
    extension_problem, parse_decision, path_safety_problems, resolve_alias,
)

ALLOWED_ACTIONS = frozenset({*TOOL_EXTENSIONS, "answer", "ask"})


class Outcome(StrEnum):
    APPROVED_FOR_REVIEW = "approved_for_review"  # may be SHOWN to a human; not authorization
    NEEDS_CLARIFICATION = "needs_clarification"  # intent uncertain -> fixed clarification
    REJECTED = "rejected"                        # structural/path failure -> fixed clarification


class IntentConcern(StrEnum):
    NEGATED_REFERENCE = "negated_reference"
    SUPERSEDED_REFERENCE = "superseded_reference"
    CONFLICTING_INSTRUCTIONS = "conflicting_instructions"
    MULTIPLE_FILE_REFERENCES = "multiple_file_references"
    RETRACTED_OR_DECLINED = "retracted_or_declined"
    ADVICE_QUESTION = "advice_question"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    NO_INSPECTION_INTENT = "no_inspection_intent"
    ANSWER_INSTEAD_OF_INSPECTION = "answer_instead_of_inspection"


@dataclass(frozen=True)
class ValidationResult:
    outcome: Outcome
    action: str | None
    tool_path: str | None
    user_reference: str | None
    structural_reasons: tuple[Reason, ...]
    intent_concerns: tuple[IntentConcern, ...]

    @property
    def ready_for_human_review(self) -> bool:
        return self.outcome is Outcome.APPROVED_FOR_REVIEW and self.action in TOOL_EXTENSIONS


# --- Layer 2 vocabulary --------------------------------------------------------
_APOS = "['’]"
_BOUNDARY = re.compile(r"[.;!?\n,]|\bbut\b|\bhowever\b|\bthen\b|—|–| - ", re.I)
_FILLER = r"(?:the|a|an|my|your|our|this|that|these|those|file|files|script|module|csv|spreadsheet|one|called|named|at|in|on|of)"
_DIRECT_NEG = re.compile(
    r"\b(?:not|except|excluding|exclude|instead\s+of|rather\s+than|other\s+than|apart\s+from|besides|skip|skipping|"
    rf"avoid|avoiding|ignore|ignoring|leave|leaving|without|nor|neither|minus)(?:\s+{_FILLER})*\s*$", re.I)
_ACCESS = (r"(?:inspect\w*|open\w*|read\w*|check\w*|look\w*|touch\w*|access\w*|analy[sz]\w*|count\w*|list\w*|"
           r"scan\w*|summari[sz]\w*|show\w*|use|uses|using|process\w*|pars(?:e|es|ed|ing)|examin\w*|view\w*|load\w*)")
_DIRECTIVE_NEG = re.compile(
    rf"(?:\b(?:do\s+not|dont|never|no\s+need\s+to|refrain\s+from|stop|must\s+not|should\s+not|cannot|can\s+not)\b|\w+n{_APOS}t\b)"
    rf"(?:\W+[\w'’]+){{0,6}}?\W+{_ACCESS}\b(?:\W+[\w'’]+){{0,5}}\W*$", re.I)
_RHETORICAL = re.compile(rf"\b(?:ca|could|wo|would|why\s+do)n{_APOS}?t\s+you\b|\bcant\s+you\b|\bwhy\s+not\b", re.I)
_POST_NEG = re.compile(
    r"^\W*(?:alone\b|untouched\b|aside\b|out\s+of\s+(?:it|this)\b"
    r"|(?:is|are|was|were|should|must|does|do|need|needs|will|would)\s+(?:not|never)\b"
    rf"|(?:isn|aren|shouldn|mustn|doesn|don|won){_APOS}t\b"
    r"|(?:should|can|must)\s+be\s+(?:ignored|skipped|excluded|left\s+alone)\b|is\s+off[-\s]limits\b)", re.I)
_RETRACTED = re.compile(
    rf"\bon\s+second\s+thoughts?\b|\bnever\s*mind\b|\bscratch\s+that\b|\bforget\s+(?:it|that|about\s+it)\b|\bcancel\b"
    rf"|\bdon{_APOS}?t\s+bother\b|\bhold\s+off\b|\bnot\s+now\b|\b(?:actually|wait)\W+(?:no|don{_APOS}?t|do\s+not|stop|never)\b"
    rf"|\bwithout\s+(?:using\s+|calling\s+|running\s+)?(?:any\s+)?tools?\b|\bno\s+tools?\b"
    rf"|\b(?:do\s+not|don{_APOS}?t)\s+(?:use|call|run|invoke)\s+(?:any\s+)?(?:tools?|functions?)\b"
    rf"|\b(?:do\s+not|don{_APOS}?t)\s+(?:do|open|inspect|read|check|touch)\s+(?:it|that|anything)\b", re.I)
_ADVICE = re.compile(
    r"\b(?:should|shall|must|do|would|could|can|may)\s+(?:i|we)\b(?!\s+(?:see|get|have|know|please))"
    r"|\bis\s+it\s+(?:worth|safe|ok|okay|a\s+good\s+idea|wise|necessary)\b|\bwhat\s+(?:would|will)\s+happen\b"
    r"|\bwhether\s+(?:to|i\s+should|we\s+should)\b|\bwould\s+it\s+be\b", re.I)
_CORRECTION = re.compile(
    rf"\bno\s+wait\b|\bi\s+meant\b|\bi\s+mean\b|\bcorrection\b|\bmake\s+that\b|\bi\s+meant\s+to\s+say\b"
    rf"|\bactually\W+(?:use|inspect|check|make\s+it|it{_APOS}?s)\b|\bsorry\W+(?:i\s+meant|it{_APOS}?s)\b", re.I)


def _mask(message: str, refs: tuple[FileReference, ...]) -> str:
    chars = list(message)
    for r in refs:
        chars[r.start:r.end] = "_" * (r.end - r.start)
    return "".join(chars)


def _clause(masked: str, start: int, end: int) -> tuple[str, str]:
    before, after = masked[:start], masked[end:]
    cuts = [b.end() for b in _BOUNDARY.finditer(before)]
    nxt = _BOUNDARY.search(after)
    return before[cuts[-1] if cuts else 0:], after[:nxt.start()] if nxt else after


@dataclass(frozen=True)
class _RefIntent:
    ref: FileReference
    negated: bool
    superseded: bool


def _reference_intents(message: str, refs: tuple[FileReference, ...]) -> tuple[_RefIntent, ...]:
    masked = _mask(message, refs)
    neutral = _RHETORICAL.sub(lambda m: " " * len(m.group(0)), masked)  # "can't you ..." is a request
    corrections = [m.end() for m in _CORRECTION.finditer(masked)]
    cut = corrections[-1] if corrections else None
    has_ref_after_cut = cut is not None and any(r.start >= cut for r in refs)
    out = []
    for r in refs:
        pre, post = _clause(neutral, r.start, r.end)
        negated = bool(_DIRECT_NEG.search(pre) or _DIRECTIVE_NEG.search(pre) or _POST_NEG.search(post))
        superseded = bool(has_ref_after_cut and r.start < cut)
        out.append(_RefIntent(r, negated, superseded))
    return tuple(out)


def _assess_intent(message: str, action: str, target: str | None, refs, analysis) -> list[IntentConcern]:
    concerns: list[IntentConcern] = []
    masked = _mask(message, refs)
    ri = _reference_intents(message, refs)
    active = [x for x in ri if not x.negated and not x.superseded]
    if target is not None:
        hits = [x for x in ri if x.ref.target == target]
        if hits and not any(x in active for x in hits):
            concerns.append(IntentConcern.SUPERSEDED_REFERENCE if all(x.superseded for x in hits)
                            else IntentConcern.NEGATED_REFERENCE)
    negated_targets = {x.ref.target for x in ri if x.negated}
    if any(x.ref.target in negated_targets for x in active):
        concerns.append(IntentConcern.CONFLICTING_INSTRUCTIONS)
    if len({x.ref.target for x in active}) > 1:
        concerns.append(IntentConcern.MULTIPLE_FILE_REFERENCES)
    if _RETRACTED.search(masked):
        concerns.append(IntentConcern.RETRACTED_OR_DECLINED)
    if _ADVICE.search(masked):
        concerns.append(IntentConcern.ADVICE_QUESTION)
    if analysis.unsupported_operation:
        concerns.append(IntentConcern.UNSUPPORTED_OPERATION)
    if action in TOOL_EXTENSIONS and action not in analysis.intents:
        concerns.append(IntentConcern.NO_INSPECTION_INTENT)
    return concerns


def _single_clear_target(message: str, refs, analysis) -> bool:
    """True when the user plainly asked to inspect exactly one valid file."""
    if any(r.unsafe for r in refs) or analysis.suspicious:
        return False
    active = [x for x in _reference_intents(message, refs) if not x.negated and not x.superseded]
    if len({x.ref.target for x in active}) != 1:
        return False
    tool = next((t for t, ext in TOOL_EXTENSIONS.items()
                 if PurePosixPath(active[0].ref.target).suffix.lower() == ext), None)
    return tool is not None and not _assess_intent(message, tool, active[0].ref.target, refs, analysis)


def _result(outcome, action, structural=(), concerns=(), tool_path=None, user_reference=None):
    return ValidationResult(outcome, action, tool_path, user_reference,
                            tuple(dict.fromkeys(structural)), tuple(dict.fromkeys(concerns)))


def validate_decision(user_message: str, decision: Mapping[str, Any] | ModelDecision) -> ValidationResult:
    """Classify a proposed decision. Never executes, repairs or substitutes anything."""
    parsed, problems = parse_decision(decision)
    if parsed is None:
        return _result(Outcome.REJECTED, None, problems)
    if not isinstance(user_message, str) or not user_message.strip() or len(user_message) > MAX_USER_CHARS:
        return _result(Outcome.REJECTED, parsed.action, [Reason.INVALID_USER_MESSAGE])
    if parsed.action not in ALLOWED_ACTIONS:
        return _result(Outcome.REJECTED, None, [Reason.UNSUPPORTED_ACTION])
    analysis = analyze_request(user_message)  # used for reference extraction and flags only
    refs = analysis.references

    # ---- Layer 1: structure and path (hard) ----
    structural: list[Reason] = []
    if parsed.action not in TOOL_EXTENSIONS:
        if parsed.path:
            structural.append(Reason.PATH_WITH_NON_TOOL_ACTION)
        if not parsed.message.strip():
            structural.append(Reason.MISSING_MESSAGE)
        if structural:
            return _result(Outcome.REJECTED, parsed.action, structural)
        if parsed.action == "answer" and _single_clear_target(user_message, refs, analysis):
            return _result(Outcome.NEEDS_CLARIFICATION, "answer", (), [IntentConcern.ANSWER_INSTEAD_OF_INSPECTION])
        return _result(Outcome.APPROVED_FOR_REVIEW, parsed.action)

    if parsed.message.strip():
        structural.append(Reason.TOOL_WITH_MESSAGE)
    safety = path_safety_problems(parsed.path)
    structural += safety
    target = resolve_alias(parsed.path)
    if target != parsed.path:
        structural += path_safety_problems(target)
    if parsed.path and not safety:
        ext = extension_problem(parsed.action, parsed.path)
        if ext:
            structural.append(ext)
    matching = [r for r in refs if r.target == target]
    if parsed.path and not matching:
        structural.append(Reason.PATH_NOT_REFERENCED)  # invented or changed
    if any(r.unsafe for r in refs):
        structural.append(Reason.UNSAFE_REFERENCE_IN_REQUEST)
    if analysis.suspicious:
        structural.append(Reason.SUSPICIOUS_INSTRUCTIONS)
    if structural:
        return _result(Outcome.REJECTED, parsed.action, structural)

    # ---- Layer 2: intent assessment (heuristic) ----
    concerns = _assess_intent(user_message, parsed.action, target, refs, analysis)
    if concerns:
        return _result(Outcome.NEEDS_CLARIFICATION, parsed.action, (), concerns)
    ref = next(x.ref for x in _reference_intents(user_message, refs)
               if x.ref.target == target and not x.negated and not x.superseded)
    return _result(Outcome.APPROVED_FOR_REVIEW, parsed.action, tool_path=target, user_reference=ref.text)
