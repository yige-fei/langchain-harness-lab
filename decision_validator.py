"""Deterministic validator for decisions proposed by the local model (Phase 4B.2).

Pure standard library: no MCP, LangChain, network or filesystem access. It never
executes anything and never repairs or replaces a decision. It answers one
question: may this proposal go on to human approval?

An APPROVED tool decision is *not* permission to execute. It still needs human
approval, and SamToolkit's own path checks remain a separate security boundary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Mapping

# --- Contract -----------------------------------------------------------------
TOOL_EXTENSIONS = {"inspect_python": ".py", "inspect_csv": ".csv"}
ALLOWED_ACTIONS = frozenset({*TOOL_EXTENSIONS, "answer", "ask"})
DECISION_FIELDS = frozenset({"action", "path", "message"})
WORKSPACE_ALIAS = "examples/"  # the authorized workspace root, as users name it from the repo

MAX_USER_CHARS = 4_000
MAX_MESSAGE_CHARS = 4_000
MAX_PATH_CHARS = 255


class Outcome(StrEnum):
    APPROVED = "approved"  # may proceed to HUMAN approval; never executed automatically
    REJECTED = "rejected"  # stop; no replacement action is invented


class Reason(StrEnum):
    # decision structure
    MALFORMED_DECISION = "malformed_decision"
    UNEXPECTED_FIELDS = "unexpected_fields"
    FIELD_TOO_LONG = "field_too_long"
    INVALID_USER_MESSAGE = "invalid_user_message"
    UNSUPPORTED_ACTION = "unsupported_action"
    MISSING_PATH = "missing_path"
    MISSING_MESSAGE = "missing_message"
    TOOL_WITH_MESSAGE = "tool_with_message"
    PATH_WITH_NON_TOOL_ACTION = "path_with_non_tool_action"
    # path syntax
    ABSOLUTE_PATH = "absolute_path"
    HOME_PATH = "home_path"
    TRAVERSAL = "traversal"
    HIDDEN_PATH = "hidden_path"
    MALFORMED_PATH = "malformed_path"
    DISALLOWED_CHARACTERS = "disallowed_characters"
    SCHEME_OR_DRIVE = "scheme_or_drive"
    BACKSLASH = "backslash"
    PATH_TOO_LONG = "path_too_long"
    UNSUPPORTED_EXTENSION = "unsupported_extension"
    EXTENSION_TOOL_MISMATCH = "extension_tool_mismatch"
    # relation to the user's request
    PATH_NOT_REFERENCED = "path_not_referenced"  # invented or changed
    NEGATED_REFERENCE = "negated_reference"
    MULTIPLE_FILE_REFERENCES = "multiple_file_references"
    UNSAFE_REFERENCE_IN_REQUEST = "unsafe_reference_in_request"
    USER_DECLINED_TOOLS = "user_declined_tools"
    SUSPICIOUS_INSTRUCTIONS = "suspicious_instructions"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    NO_INSPECTION_INTENT = "no_inspection_intent"
    ANSWER_INSTEAD_OF_INSPECTION = "answer_instead_of_inspection"


@dataclass(frozen=True)
class ModelDecision:
    action: str
    path: str
    message: str


@dataclass(frozen=True)
class FileReference:
    """A file/path reference found in the user's message."""
    text: str          # exactly as the user wrote it
    start: int
    end: int
    quoted: bool
    negated: bool
    target: str        # workspace-relative form (authorized "examples/" prefix removed)
    unsafe: bool       # absolute, traversal, home, hidden, odd characters, ...


@dataclass(frozen=True)
class RequestAnalysis:
    references: tuple[FileReference, ...]
    declined_tools: bool
    suspicious: bool
    unsupported_operation: bool
    intents: frozenset[str]  # tool names whose kind of inspection the user asked for


@dataclass(frozen=True)
class ValidationResult:
    outcome: Outcome
    action: str | None
    tool_path: str | None        # workspace-relative path for the MCP call (tools only)
    user_reference: str | None   # the exact reference the user wrote
    reasons: tuple[Reason, ...]

    @property
    def approved(self) -> bool:
        return self.outcome is Outcome.APPROVED


# --- Path rules ----------------------------------------------------------------
_SAFE_PATH_CHARS = re.compile(r"[A-Za-z0-9 _.\-/]+")


def path_safety_problems(path: str) -> tuple[Reason, ...]:
    """Syntax-only checks; stricter than (and independent of) SamToolkit's server checks."""
    if not path:
        return (Reason.MISSING_PATH,)
    found: list[Reason] = []
    if len(path) > MAX_PATH_CHARS:
        found.append(Reason.PATH_TOO_LONG)
    if "\\" in path:
        found.append(Reason.BACKSLASH)
    if ":" in path:
        found.append(Reason.SCHEME_OR_DRIVE)
    if path.startswith("/"):
        found.append(Reason.ABSOLUTE_PATH)
    if path.startswith("~"):
        found.append(Reason.HOME_PATH)
    if not _SAFE_PATH_CHARS.fullmatch(path):
        found.append(Reason.DISALLOWED_CHARACTERS)
    parts = path.split("/")
    if ".." in parts:
        found.append(Reason.TRAVERSAL)
    inner = parts[1:] if path.startswith("/") else parts
    if any(p in ("", ".") or p != p.strip() for p in inner):
        found.append(Reason.MALFORMED_PATH)
    if any(p.startswith(".") and p not in (".", "..") for p in parts):
        found.append(Reason.HIDDEN_PATH)
    return tuple(dict.fromkeys(found))


def extension_problem(action: str, path: str) -> Reason | None:
    suffix = PurePosixPath(path).suffix.lower()  # SamToolkit also compares lower-cased suffixes
    if suffix not in TOOL_EXTENSIONS.values():
        return Reason.UNSUPPORTED_EXTENSION
    if suffix != TOOL_EXTENSIONS[action]:
        return Reason.EXTENSION_TOOL_MISMATCH
    return None


def resolve_alias(path: str) -> str:
    """Map the authorized "examples/" prefix to the workspace root. Nothing else is rewritten."""
    if path.startswith(WORKSPACE_ALIAS) and len(path) > len(WORKSPACE_ALIAS):
        return path[len(WORKSPACE_ALIAS):]
    return path


# --- Reading the user's message -----------------------------------------------
_QUOTED = [
    re.compile(r'"([^"\n]{1,255})"'),
    re.compile(r"“([^”\n]{1,255})”"),
    re.compile(r"`([^`\n]{1,255})`"),
    re.compile(r"(?<![A-Za-z0-9])'([^'\n]{1,255})'(?![A-Za-z0-9])"),
    re.compile(r"(?<![A-Za-z0-9])‘([^’\n]{1,255})’(?![A-Za-z0-9])"),
]
_TOKEN = re.compile(r"[^\s\"'`“”‘’,;()<>\[\]{}|]+")
_FILE_NAME = re.compile(r".*[A-Za-z0-9_\-].*\.[A-Za-z][A-Za-z0-9]{0,9}")
_NOT_FILES = frozenset({"e.g", "i.e", "a.m", "p.m", "etc", "vs", "approx"})
_PATHISH = re.compile(r"^(?:/|~|\.\./)|/\.\.(?:/|$)|\\")

_BOUNDARY = re.compile(r"[.;!?\n,:]|\bbut\b|\bhowever\b|\bthen\b|—|–| - ", re.I)
_NEG_BEFORE = re.compile(
    r"\b(?:not|never|without|except|excluding|exclude|ignore|ignoring|skip|skipping|avoid|avoiding|"
    r"neither|nor|instead\s+of|rather\s+than|other\s+than|apart\s+from|no\s+need\s+to|dont)\b|n['’]t\b",
    re.I)
_NEG_AFTER = re.compile(
    r"^\s*(?:(?:is|are|was|were|should|must|does|do|need|needs|will|would)\s+(?:not|never)\b"
    r"|(?:isn|aren|shouldn|mustn|doesn|don|won)['’]t\b"
    r"|(?:should|can|must)\s+be\s+(?:ignored|skipped|excluded))", re.I)
_DECLINE = re.compile(
    r"\bnever\s*mind\b|\bforget\s+(?:it|that|about\s+it)\b|\bcancel\b"
    r"|\b(?:do\s+not|don['’]?t)\s+(?:use|call|run|invoke)\s+(?:any\s+)?(?:tools?|functions?)\b"
    r"|\bwithout\s+(?:using\s+|calling\s+|running\s+)?(?:any\s+)?tools?\b|\bno\s+tools?\b"
    r"|\b(?:actually|wait)\b[^.?!]*\b(?:don['’]?t|do\s+not|stop)\b"
    r"|\b(?:do\s+not|don['’]?t)\s+(?:do|open|inspect|read|check|touch)\s+(?:it|that|anything)\b", re.I)
_SUSPICIOUS = re.compile(
    r"\bignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier|your)\s+(?:instructions|rules|guidelines)\b"
    r"|\bsystem\s*:|\bvalidator\b|\bapprove[ds]?\b|\boverride\b|\bjailbreak\b|\bdeveloper\s+mode\b"
    r"|<\s*/?\s*(?:system|tool_call|tools)\s*>|\bpre-?approved\b|\bsudo\b", re.I)
_UNSUPPORTED_OP = re.compile(
    r"\b(?:delete|remove|rm|rename|move|mv|copy|cp|write|overwrite|edit|modify|change|create|append|"
    r"chmod|chown|execute|exec|run|install|upload|download|send|email|post|fix|refactor)\b", re.I)
_GENERIC_INTENT = r"inspect|examine|analy[sz]e|summari[sz]e|look\s+at|check|show|describe|overview|contents?|what['’]?s\s+in"
_INTENT = {
    "inspect_python": re.compile(
        rf"\b(?:{_GENERIC_INTENT}|functions?|def|defs|class(?:es)?|methods?|async|structure|outline|"
        rf"defines?|defined|symbols?|list)\b", re.I),
    "inspect_csv": re.compile(
        rf"\b(?:{_GENERIC_INTENT}|rows?|columns?|headers?|fields?|missing|blank|empty|null|nan|cells?|"
        rf"values?|records?|lines?|count|how\s+many|size|shape)\b", re.I),
}


def _looks_like_reference(text: str) -> bool:
    if text.lower() in _NOT_FILES:
        return False
    return bool(_FILE_NAME.fullmatch(text.rsplit("/", 1)[-1])) or bool(_PATHISH.search(text))


def _is_unsafe(text: str) -> bool:
    return bool(path_safety_problems(text)) or bool(path_safety_problems(resolve_alias(text)))


def analyze_request(message: str) -> RequestAnalysis:
    spans: list[tuple[int, int, str, bool]] = []  # (start, end, text, quoted)
    taken = [False] * len(message)
    for pattern in _QUOTED:
        for m in pattern.finditer(message):
            inner = m.group(1)
            if any(taken[m.start():m.end()]) or not _looks_like_reference(inner):
                continue
            spans.append((m.start(), m.end(), inner, True))
            taken[m.start():m.end()] = [True] * (m.end() - m.start())
    for m in _TOKEN.finditer(message):
        if any(taken[m.start():m.end()]):
            continue
        raw = m.group(0)
        text = raw.rstrip(".,:;!?")
        if text and _looks_like_reference(text):
            spans.append((m.start(), m.start() + len(text), text, False))
    spans.sort()

    # Mask references so their dots and words are not read as sentence boundaries or cues.
    masked_chars = list(message)
    for start, end, _, _ in spans:
        masked_chars[start:end] = "_" * (end - start)
    masked = "".join(masked_chars)

    refs = []
    for start, end, text, quoted in spans:
        before = masked[:start]
        cuts = [b.end() for b in _BOUNDARY.finditer(before)]
        pre = before[cuts[-1] if cuts else 0:]
        after = masked[end:]
        nxt = _BOUNDARY.search(after)
        post = after[:nxt.start()] if nxt else after
        negated = bool(_NEG_BEFORE.search(pre) or _NEG_AFTER.search(post))
        refs.append(FileReference(text, start, end, quoted, negated, resolve_alias(text), _is_unsafe(text)))

    return RequestAnalysis(
        references=tuple(refs),
        declined_tools=bool(_DECLINE.search(masked)),
        suspicious=bool(_SUSPICIOUS.search(masked)),
        unsupported_operation=bool(_UNSUPPORTED_OP.search(masked)),
        intents=frozenset(tool for tool, rx in _INTENT.items() if rx.search(masked)),
    )


# --- Decision checks -----------------------------------------------------------
def parse_decision(decision: Mapping[str, Any] | ModelDecision) -> tuple[ModelDecision | None, tuple[Reason, ...]]:
    if isinstance(decision, ModelDecision):
        decision = {"action": decision.action, "path": decision.path, "message": decision.message}
    if not isinstance(decision, Mapping):
        return None, (Reason.MALFORMED_DECISION,)
    keys = set(decision)
    if keys - DECISION_FIELDS:
        return None, (Reason.UNEXPECTED_FIELDS,)
    if keys != DECISION_FIELDS or not all(isinstance(decision[k], str) for k in DECISION_FIELDS):
        return None, (Reason.MALFORMED_DECISION,)
    if len(decision["message"]) > MAX_MESSAGE_CHARS or len(decision["action"]) > 64:
        return None, (Reason.FIELD_TOO_LONG,)
    return ModelDecision(decision["action"], decision["path"], decision["message"]), ()


def _reject(action: str | None, reasons) -> ValidationResult:
    return ValidationResult(Outcome.REJECTED, action, None, None, tuple(dict.fromkeys(reasons)))


def _inspectable_target(analysis: RequestAnalysis) -> FileReference | None:
    """The single file the user plainly asked to inspect, if there is exactly one."""
    if analysis.declined_tools or analysis.suspicious or analysis.unsupported_operation:
        return None
    if any(r.unsafe for r in analysis.references):
        return None
    active = [r for r in analysis.references if not r.negated]
    if len({r.target for r in active}) != 1:
        return None
    ref = active[0]
    tool = next((t for t, ext in TOOL_EXTENSIONS.items() if PurePosixPath(ref.target).suffix.lower() == ext), None)
    return ref if tool and tool in analysis.intents else None


def validate_decision(user_message: str, decision: Mapping[str, Any] | ModelDecision) -> ValidationResult:
    """Return APPROVED (go to human approval) or REJECTED with reasons. Never executes or repairs."""
    parsed, problems = parse_decision(decision)
    if parsed is None:
        return _reject(None, problems)
    if not isinstance(user_message, str) or not user_message.strip() or len(user_message) > MAX_USER_CHARS:
        return _reject(parsed.action, [Reason.INVALID_USER_MESSAGE])
    if parsed.action not in ALLOWED_ACTIONS:
        return _reject(None, [Reason.UNSUPPORTED_ACTION])
    analysis = analyze_request(user_message)

    if parsed.action not in TOOL_EXTENSIONS:  # answer / ask
        reasons = []
        if parsed.path:
            reasons.append(Reason.PATH_WITH_NON_TOOL_ACTION)
        if not parsed.message.strip():
            reasons.append(Reason.MISSING_MESSAGE)
        if parsed.action == "answer" and _inspectable_target(analysis):
            reasons.append(Reason.ANSWER_INSTEAD_OF_INSPECTION)  # would answer about a file without reading it
        if reasons:
            return _reject(parsed.action, reasons)
        return ValidationResult(Outcome.APPROVED, parsed.action, None, None, ())

    # inspect_python / inspect_csv
    reasons: list[Reason] = []
    if parsed.message.strip():
        reasons.append(Reason.TOOL_WITH_MESSAGE)
    safety = path_safety_problems(parsed.path)
    reasons += safety
    target = resolve_alias(parsed.path)
    if target != parsed.path:
        reasons += path_safety_problems(target)
    if parsed.path and not safety:
        ext = extension_problem(parsed.action, parsed.path)
        if ext:
            reasons.append(ext)
    active = [r for r in analysis.references if not r.negated]
    matches = [r for r in active if r.target == target]
    if parsed.path and not matches:
        negated_match = any(r.target == target for r in analysis.references if r.negated)
        reasons.append(Reason.NEGATED_REFERENCE if negated_match else Reason.PATH_NOT_REFERENCED)
    if len({r.target for r in active}) > 1:
        reasons.append(Reason.MULTIPLE_FILE_REFERENCES)
    if any(r.unsafe for r in analysis.references):
        reasons.append(Reason.UNSAFE_REFERENCE_IN_REQUEST)
    if analysis.declined_tools:
        reasons.append(Reason.USER_DECLINED_TOOLS)
    if analysis.suspicious:
        reasons.append(Reason.SUSPICIOUS_INSTRUCTIONS)
    if analysis.unsupported_operation:
        reasons.append(Reason.UNSUPPORTED_OPERATION)
    if parsed.action not in analysis.intents:
        reasons.append(Reason.NO_INSPECTION_INTENT)
    if reasons:
        return _reject(parsed.action, reasons)
    return ValidationResult(Outcome.APPROVED, parsed.action, target, matches[0].text, ())
