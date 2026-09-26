"""Text normalization, sentence segmentation and lexical helpers."""

from __future__ import annotations

import re

# Abbreviations that end with a period but do not end a sentence.
_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "dr", "mr", "mrs", "ms", "prof", "fig", "figs",
    "eq", "no", "approx", "al", "st", "jr", "sr", "inc", "ltd", "co", "u.s",
    "ca", "cf", "sec", "vol", "pp", "min", "max", "resp",
}

STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he her here hers herself him himself his
how i if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over own same
shan't she should shouldn't so some such than that that's the their theirs them
themselves then there there's these they this those through to too under until up
very was wasn't we were weren't what what's when where which while who whom why
will with won't would wouldn't you your yours yourself yourselves also however
according document source sources text tell explain describe give show please
many much does did
""".split())

CAUSAL_MARKERS = (
    "because", "due to", "caused by", "cause", "causes", "caused", "as a result",
    "resulted in", "results in", "result of", "led to", "leads to", "lead to",
    "owing to", "therefore", "thus", "consequently", "attributed to",
    "explained by", "driven by", "responsible for", "since", "so that",
    "reason", "contributed to", "contributes to",
)
COMPARISON_MARKERS = (
    "more than", "less than", "fewer than", "higher", "lower", "greater",
    "smaller", "larger", "compared", "comparison", "than", "whereas", "while",
    "versus", "vs", "difference", "differ", "similar", "same as", "exceed",
    "outperform", "increase", "decrease", "twice", "half",
)
LIMITATION_MARKERS = (
    "however", "although", "limitation", "limited", "only", "not", "cannot",
    "unclear", "uncertain", "may", "might", "caveat", "preliminary", "small sample",
    "should be interpreted", "did not", "does not", "no evidence", "unknown",
)

_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d(?:[\d,]*\d)?(?:\.\d+)?")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[.'-][A-Za-z0-9]+)*")


def clean_text(text: str) -> str:
    """Collapse whitespace, drop wiki-style reference markers and fix spacing."""
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"\[\s*(?:\d+|citation needed|edit)\s*\]", " ", text, flags=re.I)
    text = re.sub(r"\s*↑\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def normalize_key(text: str) -> str:
    """Lower-cased alphanumeric key used for duplicate detection."""
    return re.sub(r"\W+", " ", text.lower()).strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting common abbreviations and decimals."""
    text = clean_text(text)
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for match in re.finditer(r"[.!?]+[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])", text):
        end = match.end()
        candidate = text[start:end].strip()
        last_word = re.search(r"([A-Za-z.]+)[.!?]+[\"')\]]*\s*$", candidate)
        if last_word:
            token = last_word.group(1).lower().rstrip(".")
            # Single capital initials ("J. Smith") and known abbreviations.
            if token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha()):
                continue
        sentences.append(candidate)
        start = end
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def _stem(token: str) -> str:
    """Very light suffix stripping so inflections match ('nodes'/'node', 'used'/'use')."""
    if token.isdigit():
        return token
    for suffix in ("ations", "ation", "ingly", "ing", "edly", "ed", "ies", "es", "s", "ly"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    else:
        if token.endswith("ed") and len(token) >= 4:   # short forms: 'used' -> 'use'
            token = token[:-1]
    # Drop a final silent 'e' so 'node' == 'nod(es)' and 'use' == 'us(ed)'.
    if len(token) > 3 and token.endswith("e"):
        token = token[:-1]
    return token


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = [t.lower() for t in _WORD_RE.findall(text)]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [_stem(t) for t in tokens]


def content_terms(text: str) -> list[str]:
    """Distinct stemmed content terms in order of appearance."""
    seen: dict[str, None] = {}
    for token in tokenize(text):
        if len(token) > 1 or token.isdigit():
            seen.setdefault(token, None)
    return list(seen)


def extract_numbers(text: str) -> set[str]:
    """Numbers normalized for comparison ('1,200' -> '1200', '3.50' -> '3.5')."""
    numbers = set()
    for raw in _NUMBER_RE.findall(text):
        value = raw.replace(",", "").lstrip("+")
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        numbers.add(value)
    return numbers


def contains_marker(text: str, markers: tuple[str, ...]) -> bool:
    low = f" {text.lower()} "
    return any(re.search(rf"(?<![a-z]){re.escape(m)}(?![a-z])", low) for m in markers)
