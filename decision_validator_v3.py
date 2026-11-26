"""Phase 5B decision validator (v3): the v2 contract plus ONE narrowly defined `wiki_search` action.

Pure standard library: no LLM, MCP, LangChain or network. It never executes, repairs or substitutes.

Decision schema: {"action", "path", "query", "message"} (all strings).
- inspect_python / inspect_csv / answer / ask: `query` must be "" and the decision is delegated
  UNCHANGED to the frozen v2 validator (so the file-path contract is not weakened). Two extra
  intent checks are added on top: a file tool while the user also asked for a Wikipedia search, and
  an `answer` to an explicit Wikipedia-search request, both lead to clarification.
- wiki_search: `path` and `message` must be "", and `query` must be the topic of ONE explicit
  Wikipedia-search command in the user's own words (see SEARCH COMMANDS below).

Layers (same meaning as v2):
  hard rules   -> REJECTED            (schema, contradictions, unsafe/altered/invented query, injection)
  heuristics   -> NEEDS_CLARIFICATION (no explicit request, declined, advice, reported, conditional,
                                        corrections, other providers, extra operations, mixed tools)
The best outcome is APPROVED_FOR_REVIEW: a human still decides. A word such as "Wikipedia" appearing
in the message does not, by itself, authorize anything.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import decision_validator_v2 as v2
from decision_validator import _SUSPICIOUS, MAX_MESSAGE_CHARS, MAX_USER_CHARS, TOOL_EXTENSIONS, analyze_request
from decision_validator_v2 import Outcome

_SEARCH_LAB = str(Path(__file__).resolve().parent / "search_lab")
if _SEARCH_LAB not in sys.path:
    sys.path.append(_SEARCH_LAB)  # appended, never prepended: cannot shadow the standard library
import wikipedia_search as wiki  # noqa: E402  (stdlib-only adapter; used here for its query rules)

SEARCH_TOOL = "wiki_search"
FILE_TOOLS = frozenset(TOOL_EXTENSIONS)
TOOL_ACTIONS = FILE_TOOLS | {SEARCH_TOOL}
ALLOWED_ACTIONS = frozenset({*TOOL_ACTIONS, "answer", "ask"})
DECISION_FIELDS = frozenset({"action", "path", "query", "message"})
MAX_QUERY_CHARS = wiki.MAX_QUERY_CHARS  # 200
MAX_QUERY_FIELD = 1_000                 # parse limit; the 200-character rule is applied after parsing

# Names with these extensions are treated as local files: never sent to Wikipedia.
LOCAL_FILE_EXT = re.compile(
    r"\.(?:py|pyc|ipynb|csv|tsv|txt|json|jsonl|xlsx?|docx?|pptx?|pdf|env|pem|key|crt|log|db|sqlite3?|ya?ml|toml|"
    r"ini|cfg|conf|md|sh|zsh|bash|zip|tar|gz|tgz|bak|plist|pkl|parquet|xml|html?)\b", re.I)


@dataclass(frozen=True)
class ValidationResult:
    outcome: Outcome
    action: str | None
    tool_path: str | None
    query: str | None
    user_reference: str | None
    structural_reasons: tuple[str, ...]
    intent_concerns: tuple[str, ...]

    @property
    def ready_for_human_review(self) -> bool:
        return self.outcome is Outcome.APPROVED_FOR_REVIEW and self.action in TOOL_ACTIONS


def _result(outcome, action, structural=(), concerns=(), tool_path=None, query=None, user_reference=None):
    return ValidationResult(outcome, action, tool_path, query, user_reference,
                            tuple(dict.fromkeys(str(x) for x in structural)),
                            tuple(dict.fromkeys(str(x) for x in concerns)))


# --- SEARCH COMMANDS: explicit, Wikipedia-specific requests -------------------------------------
_W = r"(?:the\s+)?(?:english\s+)?wikipedia"
_TOPIC_END = r"(?P<t>(?:[^?!;\n.]|\.(?=\S)){1,300})"            # up to the end of the sentence
_TOPIC_LAZY = r"(?P<t>(?:[^?!;\n.]|\.(?=\S)){1,300}?)"          # shortest topic before "... on Wikipedia"
_COMMANDS = [
    re.compile(rf"\b(?:search|query|check)\s+(?:on\s+|in\s+|through\s+)?{_W}\s+(?:for|about|on|regarding)\s+{_TOPIC_END}", re.I),
    re.compile(rf"\b{_W}\s+search\s+(?:for|about|on)\s+{_TOPIC_END}", re.I),
    re.compile(rf"\b(?:search|look\s+up)\s+{_W}\s*:\s*{_TOPIC_END}", re.I),
    re.compile(rf"\b{_W}\s+search\s*:\s*{_TOPIC_END}", re.I),
    re.compile(rf"\b(?:use|using)\s+{_W}\s+to\s+(?:look\s+up|find(?:\s+out\s+about)?|search\s+for|research|"
               rf"learn\s+about)\s+{_TOPIC_END}", re.I),
    re.compile(rf"\bwhat\s+(?:does|do)\s+{_W}\s+(?:say|have|tell\s+(?:me|us))\s+(?:about|on)\s+{_TOPIC_END}", re.I),
    re.compile(rf"\b(?:search(?:\s+for)?|look\s+up|find|check)\s+{_TOPIC_LAZY}\s+(?:on|in|via|using|from)\s+{_W}\b", re.I),
    re.compile(rf"\blook\s+{_TOPIC_LAZY}\s+up\s+(?:on|in)\s+{_W}\b", re.I),
]
_TOPIC_CUT = re.compile(
    r",|—|–|\s-\s|\s+(?:and|then|so|to)\s+(?:then\s+)?(?:tell|summari[sz]e|give|explain|show|list|describe|let|"
    r"answer|report|return|write|say|provide|share|send|email|download|save|print|translate|compare|read|open)\b"
    r"|\s+and\s+then\b|\s+then\b|\s+please\b|\s+for\s+me\b|\s+thanks?\b|\s+thank\s+you\b"
    r"|\s+if\s+(?:you\s+can|possible|you\s+(?:don['’]?t\s+)?mind)\b", re.I)
_TOPIC_PREFIX = re.compile(
    r"^(?:(?:some|more)\s+)?(?:information|info|details|facts|an?\s+article|the\s+article|articles|the\s+page|"
    r"a\s+page|pages|the\s+entry|an\s+entry|entries)\s+(?:about|on|for|regarding|of)\s+", re.I)
_EDGE_PUNCT = "\"'“”‘’()[]<>.,;:!?`"
_NO_TOPIC = frozenset("it this that these those them they him her he she its something anything everything "
                      "stuff thing things one ones there here me you us we".split()) | {""}

# --- intent cues (checked on the message with the topic masked out) -----------------------------
_CLAUSE_BOUNDARY = re.compile(r"[.;!?\n,]|\bbut\b|\bhowever\b|—|–| - ", re.I)
_NEGATED_COMMAND = re.compile(
    r"(?:\b(?:do\s+not|dont|never|no\s+need\s+to|not|stop|without|refrain\s+from|must\s+not|should\s+not)\b"
    r"|\b\w+n['’]t\b)\W*(?:[\w'’]+\W+){0,3}$", re.I)
_SEARCH_DECLINED = re.compile(
    r"\b(?:do\s+not|don['’]?t|dont|never|no\s+need\s+to|without|stop|not)\s+(?:[\w'’]+\s+){0,3}?"
    r"(?:search\w*|look\w*\s+(?:\w+\s+)?up|googl\w*|brows\w*|us(?:e|ing)\s+wikipedia|wikipedia)\b"
    r"|\bno\s+(?:search(?:es|ing)?|look[- ]?ups?|wikipedia)\b", re.I)
_HOWTO = re.compile(r"\bhow\s+(?:do|can|could|would|should|to)\s+(?:i|we|one|you\s+)?\s*(?:search|look|find|use)\b"
                    r"|\bhow\s+to\s+(?:search|look|find|use)\b", re.I)
_REPORTED = re.compile(
    r"\b(?:told|asked|said|says|suggested|recommended|advised|instructed|ordered)\s+"
    r"(?:me|us|him|her|them|you)?\s*(?:to|that)\b|\b(?:wants?|wanted)\s+(?:me|us|him|her|them)\s+to\b"
    r"|\bsupposed\s+to\b|\bi\s+was\s+told\b", re.I)
_CONDITIONAL = re.compile(r"\b(?:if|unless|in\s+case|whenever|only\s+when)\b", re.I)
_POLITE_IF = re.compile(r"\bif\s+(?:you\s+can|possible|you\s+(?:don['’]?t\s+)?mind|that['’]?s\s+ok(?:ay)?)\b", re.I)
_OTHER_PROVIDER = re.compile(
    r"\b(?:google|bing|duckduckgo|duck\s*duck\s*go|ddg|yahoo|baidu|yandex|brave\s+search|perplexity|chatgpt|"
    r"web\s+search|internet\s+search|the\s+(?:web|internet|net)|online)\b", re.I)
_EXTRA_OPERATION = re.compile(
    r"\b(?:download|save|store|upload|send|email|e-mail|post|publish|share|print|edit|modify|delete|remove|"
    r"fetch|scrape|crawl|click|browse|bookmark)\b|\bwrite\s+(?:it|them|this|that|the\s+\w+)\s+to\b"
    r"|\bopen\s+(?:the\s+|each\s+|every\s+|all\s+)?(?:article|page|link|url)s?\b"
    r"|\bread\s+the\s+(?:full|whole|entire|complete)\b|\bfollow\s+(?:the\s+)?links?\b", re.I)
_TOPIC_INJECTION = re.compile(r"<\s*/?\s*(?:system|tool_call|tools)\s*>|\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|"
                              r"above|earlier)\s+instructions\b|\bsystem\s*:", re.I)


@dataclass(frozen=True)
class SearchCommand:
    start: int
    end: int
    topic: str          # as the user wrote it (trimmed)
    topic_start: int
    topic_end: int


def _trim_topic(raw: str, raw_start: int) -> tuple[str, int, int]:
    cut = _TOPIC_CUT.search(raw)
    text = raw[:cut.start()] if cut else raw
    lead = len(text) - len(text.lstrip())
    text = text.strip()
    start = raw_start + lead
    m = _TOPIC_PREFIX.match(text)
    if m:
        start, text = start + m.end(), text[m.end():]
    stripped = text.strip(_EDGE_PUNCT + " ")
    start += text.find(stripped) if stripped else 0
    return stripped, start, start + len(stripped)


def find_search_commands(message: str) -> list[SearchCommand]:
    """All explicit Wikipedia-search commands in the message (overlapping matches merged)."""
    found: list[SearchCommand] = []
    for rx in _COMMANDS:
        pos = 0
        while (m := rx.search(message, pos)) is not None:
            topic, ts, te = _trim_topic(m.group("t"), m.start("t"))
            # A topic at the end of the pattern stops where it was trimmed, so a second command later in
            # the same sentence ("..., then search Wikipedia for Y") is still found.
            end = max(te, m.start("t")) if m.end("t") == m.end() else m.end()
            if all(end <= c.start or m.start() >= c.end for c in found):
                found.append(SearchCommand(m.start(), end, topic, ts, te))
            pos = max(end, m.start() + 1)
    return sorted(found, key=lambda c: c.start)


def _norm_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    tokens = [t.strip(_EDGE_PUNCT) for t in text.split()]
    tokens = [t for t in tokens if t]
    if tokens and tokens[0] in ("the", "a", "an"):
        tokens = tokens[1:]
    return tokens


def _mask(message: str, spans) -> str:
    chars = list(message)
    for s, e in spans:
        chars[s:e] = "_" * (e - s)
    return "".join(chars)


def _clause_before(text: str, pos: int) -> str:
    before = text[:pos]
    cuts = [b.end() for b in _CLAUSE_BOUNDARY.finditer(before)]
    return before[cuts[-1] if cuts else 0:]


_LOCAL_FILE_TOKEN = re.compile(r"(?:^|(?<=[\s\"'(`]))[\w.-]*" + LOCAL_FILE_EXT.pattern + r"(?=$|[\s\"'),;:!?`]|\.(?:\s|$))",
                               re.I)


def _local_file_refs(text: str) -> list[str]:
    refs = [r.text for r in analyze_request(text).references
            if LOCAL_FILE_EXT.search(r.text) or "/" in r.text or "\\" in r.text]
    return refs + [m.group(0) for m in _LOCAL_FILE_TOKEN.finditer(text)]  # also dotfiles such as ".env"


def _search_concerns(message: str, commands: list[SearchCommand]) -> list[str]:
    """Heuristic intent concerns for a search; [] means one plain, explicit request."""
    concerns: list[str] = []
    if not commands:
        return ["no_explicit_search_request"]
    if len({tuple(_norm_tokens(c.topic)) for c in commands}) > 1 or len(commands) > 1:
        concerns.append("multiple_search_requests")
    rest = _mask(message, [(c.topic_start, c.topic_end) for c in commands])
    neutral = v2._RHETORICAL.sub(lambda m: " " * len(m.group(0)), rest)  # "can't you search ..." is a request
    if any(_NEGATED_COMMAND.search(_clause_before(neutral, c.start)) for c in commands) \
            or _SEARCH_DECLINED.search(neutral) or v2._RETRACTED.search(rest):
        concerns.append("search_declined")
    if v2._ADVICE.search(rest) or _HOWTO.search(rest):
        concerns.append("advice_or_howto_question")
    if _REPORTED.search(rest) or _CONDITIONAL.search(_POLITE_IF.sub(" ", rest)):
        concerns.append("reported_or_conditional_request")
    if v2._CORRECTION.search(rest):
        concerns.append("correction_in_request")
    if _OTHER_PROVIDER.search(rest):
        concerns.append("other_provider_requested")
    if _EXTRA_OPERATION.search(rest):
        concerns.append("unsupported_operation")
    if _local_file_refs(message):
        concerns.append("file_reference_in_request")
    return concerns


def clear_search_topic(message: str) -> str | None:
    """The topic when the message is one plain, explicit Wikipedia-search request; else None."""
    commands = find_search_commands(message)
    if not commands or _search_concerns(message, commands) or _SUSPICIOUS.search(
            _mask(message, [(c.topic_start, c.topic_end) for c in commands])):
        return None
    return commands[0].topic or None


def search_requested(message: str) -> bool:
    """True when the message contains a Wikipedia-search command that is not plainly declined."""
    commands = find_search_commands(message)
    return bool(commands) and "search_declined" not in _search_concerns(message, commands)


# --- schema -----------------------------------------------------------------------------------------
def parse_decision(decision: object) -> tuple[dict | None, tuple[str, ...]]:
    if not isinstance(decision, Mapping):
        return None, ("malformed_decision",)
    keys = set(decision)
    if keys - DECISION_FIELDS:
        return None, ("unexpected_fields",)
    if keys != DECISION_FIELDS or not all(isinstance(decision[k], str) for k in DECISION_FIELDS):
        return None, ("malformed_decision",)
    if (len(decision["message"]) > MAX_MESSAGE_CHARS or len(decision["action"]) > 64
            or len(decision["query"]) > MAX_QUERY_FIELD or len(decision["path"]) > 1_000):
        return None, ("field_too_long",)
    return {k: decision[k] for k in DECISION_FIELDS}, ()


def _from_v2(r: v2.ValidationResult) -> ValidationResult:
    return _result(r.outcome, r.action, r.structural_reasons, r.intent_concerns, r.tool_path, None, r.user_reference)


# --- entry point ----------------------------------------------------------------------------------------
def validate_decision(user_message: str, decision: Mapping[str, Any]) -> ValidationResult:
    """Classify a proposed decision. Never executes, repairs or substitutes anything."""
    d, problems = parse_decision(decision)
    if d is None:
        return _result(Outcome.REJECTED, None, problems)
    if not isinstance(user_message, str) or not user_message.strip() or len(user_message) > MAX_USER_CHARS:
        return _result(Outcome.REJECTED, d["action"], ["invalid_user_message"])
    if d["action"] not in ALLOWED_ACTIONS:
        return _result(Outcome.REJECTED, None, ["unsupported_action"])

    if d["action"] != SEARCH_TOOL:
        if d["query"]:
            return _result(Outcome.REJECTED, d["action"], ["query_with_non_search_action"])
        r = _from_v2(v2.validate_decision(user_message, {"action": d["action"], "path": d["path"],
                                                         "message": d["message"]}))
        if r.outcome is Outcome.APPROVED_FOR_REVIEW:
            if r.action in FILE_TOOLS and search_requested(user_message):
                return _result(Outcome.NEEDS_CLARIFICATION, r.action, (), ["multiple_actions_requested"])
            if r.action == "answer" and clear_search_topic(user_message):
                return _result(Outcome.NEEDS_CLARIFICATION, "answer", (), ["answer_instead_of_search"])
        return r

    # ---- wiki_search, layer 1: hard rules ----
    query, structural = d["query"], []
    if d["path"]:
        structural.append("path_with_search")
    if d["message"].strip():
        structural.append("message_with_search")
    problem = wiki.query_problem(query)
    if problem:
        structural.append("query_" + re.sub(r"\W+", "_", problem).strip("_"))
    if query and _local_file_refs(query):
        structural.append("file_reference_in_query")
    if query and _TOPIC_INJECTION.search(query):
        structural.append("suspicious_query")
    if query and not problem and all(t in _NO_TOPIC for t in _norm_tokens(query) or [""]):
        structural.append("query_has_no_topic")              # "it", "this", "them", ... (Phase 5C)
    analysis = analyze_request(user_message)
    if any(r.unsafe for r in analysis.references):
        structural.append("unsafe_reference_in_request")
    commands = find_search_commands(user_message)
    rest = _mask(user_message, [(c.topic_start, c.topic_end) for c in commands])
    if _SUSPICIOUS.search(rest):
        structural.append("suspicious_instructions")
    q_tokens = _norm_tokens(query)
    request_tokens = set(_norm_tokens(user_message)) | {t.strip(_EDGE_PUNCT) for t in
                                                        unicodedata.normalize("NFKC", user_message).lower().split()}
    if q_tokens and any(t not in request_tokens for t in q_tokens):
        structural.append("query_not_from_request")          # invented or unrelated terms
    elif commands and q_tokens and q_tokens != _norm_tokens(commands[0].topic):
        structural.append("query_altered")                   # narrowed, extended or reordered
    if structural:
        return _result(Outcome.REJECTED, SEARCH_TOOL, structural, query=query)

    # ---- layer 2: intent (heuristic) ----
    concerns = _search_concerns(user_message, commands)
    if concerns:
        return _result(Outcome.NEEDS_CLARIFICATION, SEARCH_TOOL, (), concerns, query=query)
    return _result(Outcome.APPROVED_FOR_REVIEW, SEARCH_TOOL, query=query, user_reference=commands[0].topic)
