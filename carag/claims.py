"""Claim extraction and decomposition.

Claims are sentence-level by default. A sentence is split further only at
clear clause boundaries (";", ", whereas", ", while", ", but", ", and <subject>")
where both halves are self-contained. Sentences that cannot be split safely are
kept whole rather than rewritten, so no claim is invented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .query import _subject_end
from .text_utils import split_sentences

_PRONOUN_START = re.compile(r"^(it|they|this|that|these|those|he|she|which|who)\b", re.I)
_SOURCE_NOUN = r"(?:evidence|sources?|documents?|text|passages?|context|reports?|study|articles?|information)"
# Statements about the sources or the answer itself rather than about the world.
_META_PATTERNS = re.compile(
    r"^(i could not find|i cannot|i can't|i don't know|i'm sorry|unable to answer|"
    rf"(?:the |these |this )?(?:provided |given )?{_SOURCE_NOUN} (?:do|does|did) not|"
    rf"there is no (?:mention|information)|no information (?:is|was) (?:provided|given)|"
    r"therefore, the answer is|note:|caveat:|in summary|sure[,!]|here is|here's|here are)", re.I)
# Attribution wrappers that are not part of the factual content.
_ATTRIBUTION = re.compile(
    r"^(?:answer\s*:\s*|"
    rf"(?:the |this |these |both |all )?(?:provided |given )?{_SOURCE_NOUN} (?:also )?"
    r"(?:states?|says|shows|indicates|mentions?|notes?|reports?|explains?|suggests?|describes?|"
    r"provides?|highlights?|emphasi[sz]es?)(?: that)?,?\s+|"
    rf"according to (?:the )?(?:provided |given )?{_SOURCE_NOUN}\s*,?\s+|"
    rf"based on (?:the )?(?:provided |given )?{_SOURCE_NOUN}\s*,?\s+|"
    r"here(?:'s| is| are) [^:]{0,80}:\s*)", re.I)
_CLAUSE_SPLIT = re.compile(r";\s+|,\s+(?:whereas|while|but)\s+|,\s+and\s+(?=(?:the|a|an|[A-Z])\w*\s)")
_CAUSAL_SPLIT = re.compile(r"\s*,?\s+\b(because|since|as a result of|due to|owing to|caused by)\b\s+", re.I)


@dataclass
class CausalParts:
    effect: str
    cause: str | None      # a verifiable clause, or None when the cause is only a noun phrase
    connective: str


def _strip_citations(text: str) -> str:
    text = re.sub(r"\s*\[(?:\d+(?:\s*,\s*\d+)*|[^\]]*#[^\]]*)\]", "", text)
    # Source references such as "(Passage 3)" or "(passages 1 and 2)".
    text = re.sub(r"\s*[\(\[](?:according to |see |from )?(?:passage|source|document|doc)s?\s*"
                  r"\d+(?:\s*(?:,|and|&|-)\s*\d+)*[\)\]]", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _as_sentence(text: str) -> str:
    text = text.strip(" ,;")
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if re.search(r"[.!?]$", text) else text + "."


def _is_factual(sentence: str) -> bool:
    words = re.findall(r"[A-Za-z0-9]+", sentence)
    if len(words) < 3 or sentence.rstrip().endswith("?"):
        return False
    return not _META_PATTERNS.search(sentence.strip())


def extract_claims(text: str) -> list[str]:
    """Split text into atomic factual claims (conservatively)."""
    claims: list[str] = []
    for sentence in split_sentences(_strip_citations(text)):
        previous = None
        while previous != sentence:   # wrappers can be nested ("Based on X, here's how...:")
            previous, sentence = sentence, _ATTRIBUTION.sub("", sentence).strip()
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
        if not _is_factual(sentence):
            continue
        pieces = [p for p in _CLAUSE_SPLIT.split(sentence) if p]
        safe = len(pieces) > 1 and all(
            len(p.split()) >= 4 and not _PRONOUN_START.match(p.strip()) for p in pieces)
        if safe:
            claims.extend(_as_sentence(p) for p in pieces)
        else:
            claims.append(_as_sentence(sentence))
    seen: set[str] = set()
    return [c for c in claims if not (c.lower() in seen or seen.add(c.lower()))]


def decompose_causal(claim: str) -> CausalParts | None:
    """'X did A because it did B' -> effect 'X did A.', cause 'X did B.'"""
    match = _CAUSAL_SPLIT.search(claim)
    if not match or match.start() == 0:
        return None
    effect = _as_sentence(claim[: match.start()])
    remainder = claim[match.end():].strip().rstrip(".")
    connective = match.group(1).lower()
    if len(effect.split()) < 3 or not remainder:
        return None
    cause: str | None = None
    if connective in ("because", "since"):
        tokens = effect.rstrip(".").split()
        subject_tokens = tokens[: max(1, _subject_end(tokens, expect_verb=True))]
        subject = " ".join(subject_tokens)
        if re.match(r"^(it|they|he|she|its|their|his|her)\b", remainder, re.I) and all(
                t.lower() in ("the", "a", "an", "this", "that", "these", "those") for t in subject_tokens):
            return CausalParts(effect=effect, cause=None, connective=connective)   # cannot resolve pronoun
        # Resolve a leading pronoun to the effect clause's subject.
        remainder = re.sub(r"^(it|they|he|she)\b", subject, remainder, flags=re.I)
        remainder = re.sub(r"^(its|their|his|her)\b", subject + "'s", remainder, flags=re.I)
        cause = _as_sentence(remainder)
    return CausalParts(effect=effect, cause=cause, connective=connective)
