"""Rule-based question analysis: intent, key terms, vagueness and premises.

This is deliberately transparent and heuristic (no model). It decides which
evidence-type signals the retriever may use, whether the question is too vague
to answer, and extracts a declarative premise from "why"/yes-no questions so the
premise itself can be verified (to catch misleading assumptions).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text_utils import content_terms

_QUESTION_NOISE = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how",
                   "happen", "happened", "thing", "stuff", "someth", "anyth", "effect", "affect",
                   "caus", "cause", "reason", "result", "way", "kind", "type", "lead", "led"}
_PRONOUNS = {"it", "this", "that", "they", "them", "these", "those", "he", "she"}
_DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "its", "their", "our", "his", "her"}
_PREDICATE_STARTERS = {
    "more", "less", "fewer", "higher", "lower", "greater", "smaller", "larger", "better",
    "worse", "not", "so", "too", "very", "faster", "slower", "longer", "shorter", "only",
}
_COMMON_VERBS = {
    "consume", "use", "increase", "decrease", "fail", "produce", "cause", "have", "get",
    "become", "show", "require", "perform", "achieve", "reduce", "drop", "rise", "fall",
    "run", "take", "make", "lead", "result", "outperform", "improve", "record", "report",
    "exceed", "need", "draw", "last", "work", "change", "reach", "stop", "start", "win",
    "lose", "grow", "decline", "differ", "contain", "include", "affect", "detect", "find",
    "measure", "generate", "cost", "go", "come", "see", "give", "keep", "remain", "operate",
    "overheat", "crash", "collapse", "form", "orbit", "support", "prevent", "occur", "happen",
}
_PARTICIPLES = {"made", "built", "kept", "found", "put", "set", "run", "done", "held", "lost", "sent",
                "left", "cut", "shut", "told", "sold", "brought", "bought", "caught", "taught"}
_IRREGULAR_PAST = {
    "have": "had", "get": "got", "become": "became", "rise": "rose", "fall": "fell",
    "run": "ran", "take": "took", "make": "made", "lead": "led", "draw": "drew",
    "win": "won", "lose": "lost", "grow": "grew", "find": "found", "go": "went",
    "come": "came", "see": "saw", "give": "gave", "keep": "kept", "cost": "cost",
    "do": "did", "be": "was", "drop": "dropped", "stop": "stopped", "begin": "began",
}
_THIRD_PERSON_IRREGULAR = {"have": "has", "do": "does", "go": "goes"}


@dataclass
class QueryAnalysis:
    question: str
    key_terms: list[str]
    intents: set[str] = field(default_factory=set)   # causal, comparison, numeric, limitation, yes_no
    is_vague: bool = False
    vague_reason: str = ""
    premise: str | None = None   # declarative statement the question presupposes or asks

    @property
    def is_yes_no(self) -> bool:
        return "yes_no" in self.intents


def _past_tense(verb: str) -> str:
    low = verb.lower()
    if low in _IRREGULAR_PAST:
        return _IRREGULAR_PAST[low]
    if low.endswith("e"):
        return low + "d"
    if re.search(r"[^aeiou]y$", low):
        return low[:-1] + "ied"
    return low + "ed"


def _third_person(verb: str) -> str:
    low = verb.lower()
    if low in _THIRD_PERSON_IRREGULAR:
        return _THIRD_PERSON_IRREGULAR[low]
    if re.search(r"(s|sh|ch|x|z|o)$", low):
        return low + "es"
    if re.search(r"[^aeiou]y$", low):
        return low[:-1] + "ies"
    return low + "s"


def _subject_end(tokens: list[str], expect_verb: bool) -> int:
    """Index where the predicate starts (heuristic noun-phrase boundary)."""
    if not tokens:
        return 0
    i = 0
    if tokens[0][:1].isupper() or tokens[0][:1].isdigit():
        while i < len(tokens) and (tokens[i][:1].isupper() or tokens[i][:1].isdigit()):
            i += 1
        return i
    if tokens[0].lower() in _DETERMINERS:
        i = 1
    limit = min(len(tokens), i + 4)
    j = i
    while j < limit:
        low = tokens[j].lower()
        if j > i and (low in _PREDICATE_STARTERS or (expect_verb and low in _COMMON_VERBS)):
            return j
        # After is/was/are/were the predicate often starts with a participle ("chosen", "excluded").
        if j > i and not expect_verb and (re.search(r"[a-z]{2,}(ed|en)$", low) or low in _PARTICIPLES):
            return j
        j += 1
    # Fall back: subject is determiner + one word.
    return min(len(tokens), i + 1)


def _declarative(question: str) -> str | None:
    """Turn 'Why did X consume more?' / 'Did X consume more?' into 'X consumed more.'"""
    q = question.strip().rstrip("?").strip()
    match = re.match(r"^(?:why|how come)\s+(.*)$", q, flags=re.I)
    body = match.group(1) if match else q
    aux_match = re.match(r"^(did|does|do|is|are|was|were|has|have|had|can|could|will|would)\s+(.+)$",
                         body, flags=re.I)
    if not aux_match:
        return None
    aux, rest = aux_match.group(1).lower(), aux_match.group(2)
    tokens = rest.split()
    if len(tokens) < 2:
        return None
    split = _subject_end(tokens, expect_verb=aux in ("did", "does", "do"))
    if split <= 0 or split >= len(tokens):
        return None
    subject, predicate = tokens[:split], tokens[split:]
    if aux == "did":
        predicate = [_past_tense(predicate[0])] + predicate[1:]
    elif aux == "does":
        predicate = [_third_person(predicate[0])] + predicate[1:]
    elif aux != "do":
        predicate = [aux] + predicate
    sentence = " ".join(subject + predicate)
    return sentence[0].upper() + sentence[1:] + "."


def analyze_question(question: str) -> QueryAnalysis:
    q = question.strip()
    low = q.lower()
    intents: set[str] = set()
    if re.search(r"\b(why|how come|what (caused|causes|led to|leads to|made|is the reason|was the reason|explains?)|"
                 r"reason|cause[sd]?|because|due to|result(ed)? in|lead to|led to|effect of|impact of|"
                 r"how did .* (affect|influence|change))\b", low):
        intents.add("causal")
    if re.search(r"\b(compare|comparison|compared|differ(ence|ent)?|versus|vs\.?|than|between .* and|"
                 r"which .* (more|less|higher|lower|better|worse|faster|slower)|more|less|higher|lower)\b", low):
        intents.add("comparison")
    if re.search(r"\b(how (many|much|long|often|far|fast|large|high)|what (percentage|percent|number|amount|value|rate|"
                 r"frequency|temperature|size|duration|proportion)|how many|what was the .* (rate|value|count|level))\b", low):
        intents.add("numeric")
    if re.search(r"\b(limitation|limitations|caveat|weakness|drawback|shortcoming|uncertain|reliab|generaliz)", low):
        intents.add("limitation")
    if re.match(r"^(is|are|was|were|did|does|do|has|have|had|can|could|will|would)\b", low):
        intents.add("yes_no")

    terms = [t for t in content_terms(q) if t not in _QUESTION_NOISE]
    analysis = QueryAnalysis(question=q, key_terms=terms, intents=intents)

    words = re.findall(r"[a-z]+", low)
    if not terms:
        analysis.is_vague = True
        analysis.vague_reason = "The question does not name any specific subject to look up."
    elif len(terms) == 1 and any(w in _PRONOUNS for w in words):
        analysis.is_vague = True
        analysis.vague_reason = (
            "The question refers to something ('it', 'this', 'that'...) without naming it, "
            "so the intended subject or outcome is ambiguous."
        )

    if "yes_no" in intents or low.startswith(("why ", "how come ")):
        analysis.premise = _declarative(q)
    return analysis
