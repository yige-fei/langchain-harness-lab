"""Phase 5C: grounding for Wikipedia search answers. Standard library only; deterministic.

Two pieces:

1. evidence_answer(): the DEFAULT answer mode. It generates no factual claims at all. It quotes each
   returned snippet next to its own title and URL, labels the evidence as incomplete and says when no
   snippet contains words from the question. Disagreeing snippets are never merged or resolved: each
   stays next to its own source.
   External text is treated as untrusted and defanged for display (markup and links neutralized).

2. verify_answer(): an OPTIONAL check for a model-written answer (model mode). It is lexical and
   deterministic - it never asks a model whether the model was right. Every sentence must be supported
   by ONE specific snippet (plus its title):
     - every number and every capitalized word in the sentence must appear in that snippet;
     - at least SUPPORT_RATIO of the sentence's other content words must appear in it;
     - a negation in the sentence must also appear in the snippet;
     - if the sentence cites a URL, the support must come from THAT result (a URL is a pointer, never
       proof); the whole answer must cite at least one result URL.
   Any unsupported sentence rejects the whole answer, which is then replaced by evidence_answer().
   Limits: it checks wording, not meaning. Paraphrases are rejected (safe, less fluent); a sentence
   that reuses a snippet's words in a misleading order can still pass.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

SUPPORT_RATIO = 0.8
MAX_QUOTE_CHARS = 500

STOP = frozenset("""a an the and or but if of to in on at by for with from as into onto than then that this these those
it its it's is are was were be been being am do does did done has have had having can could may might must shall should
will would also about after before over under between through during while which who whom whose what when where why how
there here such some any each every both either more most much many very so too just only own same other another
i you he she we they me him her us them my your his our their one ones s""".split())
META = frozenset("""based short search snippet snippets result results wikipedia according source sources article articles
insufficient answer question provided information mention mentions mentioned state states stated say says said list
listed fragment fragments excerpt excerpts full complete available given retrieved returned shown note however
appear appears seem seems describe describes described refer refers referred sufficient enough unclear""".split())
NEGATIONS = frozenset({"not", "no", "never", "without", "none", "neither", "nor", "cannot", "n't"})
COMMAND_WORDS = frozenset("""search look find check use using please tell me summarize summarise give explain show
quick do wiki up""".split())

_URL = re.compile(r"https?://[^\s<>\"'`]+|(?<![\w/.])(?:[\w-]+\.)*wikipedia\.org/[^\s<>\"'`]*", re.I)
_WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(\[])")
_TRAILING = ".,;:!?)]}'\"*_"


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def stem(word: str) -> str:
    w = word.lower().replace("’", "'")
    if w.endswith("'s"):
        w = w[:-2]
    if w.endswith("n't"):
        return "n't"
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def word_set(text: str) -> set[str]:
    return {stem(w) for w in _WORD.findall(_norm(text))}


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(_norm(text))}


def question_terms(request: str) -> set[str]:
    return {stem(w) for w in _WORD.findall(_norm(request))
            if w.lower() not in STOP | META | COMMAND_WORDS and len(w) > 1}


# --- display safety for untrusted text --------------------------------------------------------------
def defang(text: str) -> str:
    """Neutralize markup and links in untrusted text so a later UI cannot render them as active content."""
    text = text.replace("<", "‹").replace(">", "›").replace("`", "'")
    text = re.sub(r"(?i)\b(https?|ftp|file|javascript|data)\s*:", lambda m: f"{m.group(1)}[:]", text)
    text = re.sub(r"\[([^\]]*)\]\(", r"[\1] (", text)  # break markdown links
    return text


# --- deterministic evidence answer (default) ------------------------------------------------------------
def evidence_answer(request: str, query: str, results: list[dict]) -> tuple[str, dict]:
    """A source-by-source quotation of the returned snippets. Generates no factual claims."""
    terms = question_terms(request)
    rows, relevant = [], []
    for i, r in enumerate(results, 1):
        matched = sorted(terms & word_set(f"{r['title']} {r['snippet']}"))
        (relevant if matched else rows).append((i, r, matched))
    lines = [f'Wikipedia search results for "{defang(query)}" ({len(results)} result(s)). These are short search '
             "snippets, not full articles: they are quoted exactly as returned, not verified, and may be incomplete "
             "or disagree with each other. No conclusions have been added."]
    if not relevant:
        lines.append("None of the snippets contain words from your question, so they probably do not answer it. "
                     "The search results are insufficient.")
    else:
        lines.append(f"{len(relevant)} of {len(results)} snippet(s) contain words from your question (a word "
                     "match only: it does not show that they answer it).")
    for title, group in (("Matching your question:", relevant), ("Other results (no words from your question):", rows)):
        if not group:
            continue
        lines.append("")
        lines.append(title)
        for i, r, matched in group:
            snippet = defang(r["snippet"][:MAX_QUOTE_CHARS]) or "(no snippet returned)"
            lines.append(f"[{i}] {defang(r['title'])}: {r['url']}")
            lines.append(f'    "{snippet}"')
    info = {"relevant": [i for i, _, _ in relevant], "unmatched": [i for i, _, _ in rows],
            "question_terms": sorted(terms)}
    return "\n".join(lines), info


# --- verifier for model-written answers ------------------------------------------------------------------
@dataclass(frozen=True)
class SentenceCheck:
    sentence: str
    supported: bool
    source: int | None      # 1-based result index that supports it
    reason: str


def _cited(sentence: str, urls: list[str]) -> set[int]:
    found = set()
    for m in _URL.finditer(sentence):
        cand = m.group(0)
        while cand and cand not in urls and cand[-1] in _TRAILING:
            cand = cand[:-1]
        if cand in urls:
            found.add(urls.index(cand) + 1)
        else:
            found.add(-1)  # cites something that is not a result
    return found


def check_sentence(sentence: str, results: list[dict]) -> SentenceCheck:
    urls = [r["url"] for r in results]
    cited = _cited(sentence, urls)
    if -1 in cited:
        return SentenceCheck(sentence, False, None, "cites a URL that is not a result")
    bare = _URL.sub(" ", sentence)
    raw_words = _WORD.findall(_norm(bare))
    content = [w for w in raw_words if w.lower() not in STOP and stem(w) not in META and len(w) > 1]
    nums = numbers(bare)
    if not content and not nums:
        return SentenceCheck(sentence, True, None, "no factual content")
    capitalized = {stem(w) for i, w in enumerate(raw_words) if w[:1].isupper() and w.lower() not in STOP
                   and stem(w) not in META}
    negated = any(stem(w) in NEGATIONS or w.lower() in NEGATIONS for w in raw_words)
    words = {stem(w) for w in content}
    best: tuple[float, int, str] = (-1.0, 0, "no candidate source")
    for idx in (sorted(cited) or range(1, len(results) + 1)):
        r = results[idx - 1]
        evidence = f"{r['title']} {r['snippet']}"
        have, have_nums = word_set(evidence), numbers(evidence)
        missing_nums = nums - have_nums
        missing_caps = capitalized - have
        ordinary = words - capitalized - {n for n in words if n.isdigit()}
        ratio = len(ordinary & have) / len(ordinary) if ordinary else 1.0
        if missing_nums:
            reason = f"number(s) not in source: {', '.join(sorted(missing_nums))}"
        elif missing_caps:
            reason = f"name(s) not in source: {', '.join(sorted(missing_caps))}"
        elif negated and not (have & NEGATIONS):
            reason = "negation not in source"
        elif ratio < SUPPORT_RATIO:
            reason = f"only {ratio:.0%} of its words are in the source"
        else:
            return SentenceCheck(sentence, True, idx, "supported")
        score = ratio - len(missing_nums) - len(missing_caps)
        if score > best[0]:
            best = (score, idx, reason)
    return SentenceCheck(sentence, False, None, best[2] + (" (the cited source)" if cited else ""))


def split_sentences(text: str) -> list[str]:
    protected = text
    for m in _URL.finditer(text):  # do not split inside URLs
        protected = protected.replace(m.group(0), m.group(0).replace(". ", ".\x00"))
    return [s.replace("\x00", " ").strip() for s in _SENTENCE.split(protected.strip()) if s.strip()]


def verify_answer(text: str, results: list[dict]) -> tuple[bool, list[SentenceCheck], list[str]]:
    """(accepted, per-sentence checks, answer-level problems)."""
    checks = [check_sentence(s, results) for s in split_sentences(text)]
    problems = []
    if not checks:
        problems.append("empty answer")
    if not any(_cited(c.sentence, [r["url"] for r in results]) - {-1} for c in checks):
        problems.append("no result URL is cited")
    if any(not c.supported for c in checks):
        problems.append(f"{sum(not c.supported for c in checks)} of {len(checks)} sentence(s) not supported")
    return not problems, checks, problems
