import re
import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Load pretrained models
print("Loading semantic retrieval model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading NLI verification model...")
nli_model = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-small"
)

SIMILARITY_THRESHOLD = 0.50


# Clean extracted text and reject navigation, tables, captions, and fragments.
BOILERPLATE_PATTERNS = (
    r"^from wikipedia", r"^jump to", r"^for (the|other) uses", r"^contents$",
    r"^references?$", r"^external links$", r"^see also$", r"^navigation menu$",
    r"^this article is about", r"^edit$", r"^citation needed$", r"^retrieved\s",
    r"^archived from", r"^isbn\b", r"^doi\s*:", r"^issn\b",
    r"^main article:", r"^geological periods$", r"^natural history$",
    r"^this article incorporates", r"^↑", r"^\[\s*\d+\s*\]",
    r"^formation of .* by the action of", r"^special issue:",
)

def clean_text(text):
    text = re.sub(r"\[\s*(?:\d+|citation needed)\s*\]", " ", text, flags=re.I)
    text = re.sub(r"\s*↑\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()

def is_candidate_sentence(sentence):
    text = clean_text(sentence)
    low = text.lower().strip(" .,:;—-")

    # Keep candidates compact and sentence-like.
    if not 45 <= len(text) <= 320:
        return False
    if not re.search(r"[.!?]$", text):
        return False
    if any(re.search(pattern, low) for pattern in BOILERPLATE_PATTERNS):
        return False
    if "|" in text or "jump to content" in low:
        return False

    # Reject table rows/statistics and headings accidentally joined to prose.
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(words) < 8:
        return False
    if len(re.findall(r"\d", text)) > 18:
        return False
    if re.search(r"(min\s+mean\s+max|composition by volume|surface pressure|apparent magnitude)", low):
        return False
    if re.match(r"^(mars\s+)?(image|figure|file|thumbnail)\b", low):
        return False

    # A factual candidate should contain a verb-like word and not be a list fragment.
    if not re.search(r"\b(is|are|was|were|has|have|had|contains|consists|retains|forms|formed|orbits|includes|reaches|ranges|lies|occur|occurs|can|may|could|shows|provides|produces|remains|becomes|experiences|hosts|isn't|doesn't)\b", low):
        return False
    return True

def extract_candidate_claims(documents, limit=8):
    """Extract compact factual sentences while discarding citation and reference debris."""
    seen = set()
    candidates = []

    for chunk in documents:
        text = re.sub(r"\[(?:Page \d+|URL https?://\S+?)\]\s*", "", chunk)
        # Remove inline reference markers before sentence segmentation; otherwise
        # markers such as [67] can cause adjacent sentences to be merged.
        text = re.sub(r"\[\s*(?:\d+|citation needed)\s*\]", " ", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip()

        for raw in re.split(r"(?<=[.!?])\s+", text):
            sentence = clean_text(raw)
            low = sentence.lower()
            key = re.sub(r"\W+", " ", low).strip()

            # Reject bibliography/citation entries and reference-list boilerplate.
            if (
                re.search(r"\b(doi|isbn|issn)\b", low)
                or re.search(r"\b(https?://|pdf\)|special issue:|et al\.)", low)
                or re.match(r"^\s*(?:↑|references?\b|bibliography\b)", low)
                or "this article incorporates text" in low
                or re.search(r"\(\s*\d{1,2}\s+[a-z]+\s+\d{4}\s*\)", low)
            ):
                continue

            if not key or key in seen or not is_candidate_sentence(sentence):
                continue
            seen.add(key)
            candidates.append({"claim": sentence, "support_score": 0.0})

    # Favor readable factual statements of moderate length.
    candidates.sort(key=lambda item: (abs(len(item["claim"]) - 150), len(item["claim"])))
    return candidates[:limit]

# PDF ingestion: extract text and split it into overlapping chunks
def load_pdf_chunks(pdf_path, sentences_per_chunk=3, overlap=1):
    reader = PdfReader(pdf_path)

    sentences = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        # Split extracted text into sentence-like units
        parts = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text))

        for part in parts:
            part = part.strip()

            if part:
                sentences.append(f"[Page {page_number}] {part}")

    if not sentences:
        raise ValueError(
            "No extractable text found. The PDF may be scanned or image-based."
        )

    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be positive.")

    if overlap < 0 or overlap >= sentences_per_chunk:
        raise ValueError("overlap must be non-negative and smaller than chunk size.")

    chunks = []
    step = sentences_per_chunk - overlap

    for start_index in range(0, len(sentences), step):
        chunk = sentences[start_index:start_index + sentences_per_chunk]

        if not chunk:
            break

        chunks.append(" ".join(chunk))

        if start_index + sentences_per_chunk >= len(sentences):
            break

    print(f"Extracted {len(sentences)} sentence-like units from the PDF.")
    print(f"Created {len(chunks)} sentence-based chunks.")

    return chunks


# Website ingestion: extract text and split it into chunks
def load_website_chunks(url, sentences_per_chunk=3, overlap=1):
    parsed_url = urlparse(url)

    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        raise ValueError("Please provide a valid HTTP or HTTPS URL.")

    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ClaimAwareRAG/1.0; educational research)"}
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "button", "noscript"]):
        tag.decompose()

    # Remove common reference, bibliography, and navigation containers.
    for selector in (
        ".reflist", ".references", ".mw-references-wrap", ".navbox",
        ".metadata", ".catlinks", ".printfooter", ".mw-editsection",
    ):
        for tag in soup.select(selector):
            tag.decompose()

    # Prefer the main article body when available; fall back to page text.
    main = soup.find("article") or soup.find("main") or soup.find(id="mw-content-text") or soup.body or soup
    text = main.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)

    sentences = [
        f"[URL {url}] {clean_text(part)}"
        for part in re.split(r"(?<=[.!?])\s+", text)
        if clean_text(part)
    ]

    if not sentences:
        raise ValueError("No extractable text found on the webpage.")

    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be positive.")

    if overlap < 0 or overlap >= sentences_per_chunk:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")

    chunks = []
    step = sentences_per_chunk - overlap

    for start in range(0, len(sentences), step):
        chunk = sentences[start:start + sentences_per_chunk]

        if not chunk:
            break

        chunks.append(" ".join(chunk))

        if start + sentences_per_chunk >= len(sentences):
            break

    print(f"Extracted {len(sentences)} sentence-like units from the webpage.")
    print(f"Created {len(chunks)} sentence-based chunks.")

    return chunks


# Step 1: Identify evidence gaps
def identify_evidence_gaps(claims):
    for claim in claims:
        score = claim["support_score"]

        if score >= 0.8:
            claim["status"] = "Supported"
        elif score >= 0.4:
            claim["status"] = "Partially Supported"
        else:
            claim["status"] = "Unsupported"

        claim["evidence_gap"] = round(1 - score, 2)

    return claims


# Step 2: Allocate retrieval budget adaptively
def allocate_budget(claims, budget):
    """Rank unresolved claims using gap plus transparent linguistic uncertainty cues."""
    candidates = [claim for claim in claims if claim["status"] != "Supported"]

    hedge_terms = ("possibly", "possible", "may", "might", "could", "likely",
                   "perhaps", "suggests", "appears", "reportedly", "it is believed")
    for claim in candidates:
        text = claim["claim"].lower()
        gap = claim["evidence_gap"]
        hedge_count = sum(term in text for term in hedge_terms)
        # This is a heuristic priority, not a calibrated probability.
        uncertainty = min(1.0, 0.35 + 0.15 * hedge_count +
                          (0.15 if len(text) > 220 else 0.0))
        claim["uncertainty"] = round(uncertainty, 2)
        claim["adaptive_priority"] = round(0.7 * gap + 0.3 * uncertainty, 3)

    return sorted(candidates, key=lambda claim: claim["adaptive_priority"], reverse=True)[:budget]


# Step 3: Semantic evidence retrieval

def normalize_for_match(text):
    text = re.sub(r"\[(?:Page \d+|URL https?://\S+?)\]\s*", "", text)
    return re.sub(r"\W+", " ", text.lower()).strip()

def retrieve_evidence(claim, documents, top_k=2):
    if not documents:
        return []

    # Avoid verifying a claim using the same sentence copied from the source.
    claim_key = normalize_for_match(claim)
    filtered_documents = []
    for document in documents:
        body = re.sub(r"\[(?:Page \d+|URL https?://\S+?)\]\s*", "", document)
        parts = re.split(r"(?<=[.!?])\s+", body)
        kept = [part.strip() for part in parts
                if part.strip() and normalize_for_match(part) != claim_key]
        if kept:
            prefix = re.search(r"\[(?:Page \d+|URL https?://\S+?)\]", document)
            marker = prefix.group(0) + " " if prefix else ""
            filtered_documents.append(marker + " ".join(kept))
    documents = filtered_documents
    if not documents:
        return []

    claim_embedding = model.encode(
        claim,
        convert_to_tensor=True
    )

    document_embeddings = model.encode(
        documents,
        convert_to_tensor=True
    )

    scores = util.cos_sim(
        claim_embedding,
        document_embeddings
    )[0]

    ranked_indices = scores.argsort(descending=True)

    return [
        {
            "document": documents[index],
            "score": round(scores[index].item(), 4)
        }
        for index in ranked_indices[:top_k]
    ]



# Find an exact source sentence for claims extracted from the ingested source.
# This preserves a traceable citation when retrieval removes the claim sentence
# to avoid using it as independent NLI evidence.
def find_source_sentence_citation(claim, documents, source):
    claim_key = normalize_for_match(claim)
    for document in documents:
        page_match = re.search(r"\[Page (\d+)\]", document)
        url_match = re.search(r"\[URL (https?://\S+?)\]", document)
        page = page_match.group(1) if page_match else ("Webpage" if url_match else "Unknown")

        body = re.sub(r"\[(?:Page \d+|URL https?://\S+?)\]\s*", "", document)
        for sentence in re.split(r"(?<=[.!?])\s+", body):
            sentence = sentence.strip()
            if sentence and normalize_for_match(sentence) == claim_key:
                return {
                    "source": source,
                    "page": page,
                    "evidence": sentence,
                    "verified": True,
                    "label": "direct_source_match",
                    "confidence": 1.0,
                }
    return None


# Extract the most relevant sentence for a precise citation
def get_precise_citation(
    claim, evidence, source, expected_label="entailment"
):
    page_match = re.search(r"\[Page (\d+)\]", evidence)
    url_match = re.search(r"\[URL (https?://\S+?)\]", evidence)

    page_number = (
        page_match.group(1) if page_match
        else "Webpage" if url_match
        else "Unknown"
    )

    text = re.sub(r"\[(?:Page \d+|URL https?://\S+?)\]\s*", "", evidence)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]

    if not sentences:
        return None

    embeddings = model.encode(
        [claim] + sentences,
        convert_to_tensor=True
    )

    scores = util.cos_sim(
        embeddings[0],
        embeddings[1:]
    )[0]

    ranked_indices = scores.argsort(descending=True)

    for index in ranked_indices:
        sentence = sentences[int(index)]

        result = nli_model({
            "text": sentence,
            "text_pair": claim
        })

        if isinstance(result, list):
            result = result[0]

        label = result["label"].lower()
        confidence = result["score"]

        if label.startswith("label_"):
            label_id = int(label.split("_")[1])
            id2label = nli_model.model.config.id2label
            label = id2label.get(label_id, label).lower()

        if (
            expected_label in label
            and confidence >= 0.70
        ):
            return {
                "source": source,
                "page": page_number,
                "evidence": sentence,
                "verified": True,
                "label": label,
                "confidence": round(confidence, 4)
            }

    return None


# Step 4: NLI-based claim verification
def verify_claim(claim, evidence):
    result = nli_model(
        {
            "text": evidence,
            "text_pair": claim
        }
    )

    if isinstance(result, list):
        result = result[0]

    label = result["label"].lower()
    confidence = result["score"]

    if label.startswith("label_"):
        label_id = int(label.split("_")[1])
        id2label = nli_model.model.config.id2label
        label = id2label.get(label_id, label).lower()

    print(f"NLI Label: {label}")
    print(f"NLI Confidence: {confidence:.4f}")

    if confidence < 0.70:
        status = "INSUFFICIENT"
    elif "entailment" in label:
        status = "SUPPORTED"
    elif "contradiction" in label:
        status = "CONTRADICTED"
    else:
        status = "INSUFFICIENT"

    return {
        "status": status,
        "label": label,
        "confidence": confidence
    }


# Step 5: Revise the answer
def revise_answer(claims, verification_results, claim_citations):
    """Only publish supported claims with verified citations; track missing citations separately."""
    revised_claims = []
    flagged_claims = []
    citation_missing_claims = []

    for claim in claims:
        claim_text = claim["claim"]
        result = verification_results.get(claim_text, "INSUFFICIENT")

        if isinstance(result, dict):
            result = result.get("status", "INSUFFICIENT")

        citation = claim_citations.get(claim_text)
        citation_is_verified = bool(citation and citation.get("verified") is True)

        if result == "SUPPORTED" and citation_is_verified:
            revised_claims.append(claim_text)
        elif result == "SUPPORTED":
            citation_missing_claims.append(claim_text)
            print(f"Supported by retrieved context, but no sentence-level citation was verified: {claim_text}")
        elif result == "CONTRADICTED":
            print(f"Removed contradicted claim: {claim_text}")
        else:
            flagged_claims.append(claim_text)
            print(f"Flagged for insufficient evidence: {claim_text}")

    return revised_claims, flagged_claims, citation_missing_claims


# Claims are extracted from the ingested source inside main().

# PDF documents are loaded from the command line inside main.

# Retrieval budget
budget = 3


# Main execution
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Claim-Aware Adaptive RAG with PDF or website ingestion"
    )
    parser.add_argument(
        "source",
        help="Path to a PDF file or a website URL"
    )
    args = parser.parse_args()

    if args.source.startswith(("http://", "https://")):
        documents = load_website_chunks(args.source)
    else:
        documents = load_pdf_chunks(args.source)

    claims = extract_candidate_claims(documents, limit=8)
    if not claims:
        raise ValueError(
            "No suitable factual sentences were extracted. Try a text-based PDF or a webpage with a readable article body."
        )
    print(f"\nGenerated {len(claims)} source-grounded candidate claims.")
    for i, item in enumerate(claims, start=1):
        print(f"  {i}. {item['claim']}")

    claims = identify_evidence_gaps(claims)

    selected = allocate_budget(claims, budget)

    print("\n=== Evidence Gap Analysis ===")
    for claim in claims:
        print(
            f'{claim["claim"]}: {claim["status"]}, '
            f'Gap = {claim["evidence_gap"]}'
        )

    print("\n=== Adaptive Priority Scores ===")
    for claim in claims:
        if claim["status"] != "Supported":
            print(
                f'{claim["claim"]}: '
                f'Uncertainty = {claim["uncertainty"]}, '
                f'Priority = {claim["adaptive_priority"]}'
            )

    print("\n=== Selected Claims for Retrieval ===")
    for claim in selected:
        print(claim["claim"])

    print("\n=== Dynamic Iterative Evidence Verification ===")

    verification_results = {}
    claim_citations = {}

    # Retrieve citations for claims initially marked as supported.
    for claim in claims:
        if claim["status"] == "Supported":
            citation_results = retrieve_evidence(
                claim["claim"],
                documents,
                top_k=1
            )

            if citation_results:
                claim_citations[claim["claim"]] = get_precise_citation(
                    claim["claim"],
                    citation_results[0]["document"],
                    args.source,
                    expected_label="entailment"
                )

    # Shared budget with adaptive claim scheduling
    retrieval_budget = 6
    max_attempts_per_claim = 2
    retrievals_used = 0

    # Track retrieval progress separately for each claim
    claim_states = {
        claim["claim"]: {
            "claim": claim,
            "remaining_documents": documents.copy(),
            "attempts": 0,
            "result": "INSUFFICIENT",
            "best_support_confidence": 0.0,
            "total_evidence_gain": 0.0,
            "base_priority": claim["adaptive_priority"],
            "dynamic_priority": claim["adaptive_priority"],
            "low_gain_streak": 0,
            "resolved": False,
        }
        for claim in selected
    }

    while retrievals_used < retrieval_budget:
        active = [
            state for state in claim_states.values()
            if not state["resolved"] and state["remaining_documents"]
            and state["attempts"] < max_attempts_per_claim
        ]

        if not active:
            break

        # Favor high-priority claims while distributing retrieval attempts
        # across unresolved claims.
        state = max(
            active,
            key=lambda item: (
                item["dynamic_priority"]
                / (1 + 0.8 * item["attempts"])
                / (1 + 0.75 * item["low_gain_streak"])
            )
        )

        claim_text = state["claim"]["claim"]
        state["attempts"] += 1
        retrievals_used += 1

        print(f"\nClaim: {claim_text}")
        print(
            f"Adaptive priority: "
            f"{state['claim']['adaptive_priority']:.3f}"
        )
        print(
            f"Attempt: {state['attempts']} | "
            f"Shared budget remaining: "
            f"{retrieval_budget - retrievals_used}"
        )

        evidence_list = retrieve_evidence(
            claim_text,
            state["remaining_documents"],
            top_k=1
        )

        if not evidence_list:
            state["resolved"] = True
            continue

        item = evidence_list[0]
        evidence = item["document"]
        similarity = item["score"]

        # retrieve_evidence() may return a filtered version of a source chunk
        # (with the claim sentence removed), so it may not equal an item in
        # remaining_documents. Remove the originating chunk by matching the
        # evidence text against its normalized content; if no match is found,
        # safely continue without crashing.
        evidence_key = normalize_for_match(evidence)
        matched_index = None
        for doc_index, original_doc in enumerate(state["remaining_documents"]):
            original_key = normalize_for_match(original_doc)
            if evidence_key and (evidence_key in original_key or
                                 original_key in evidence_key):
                matched_index = doc_index
                break
        if matched_index is not None:
            state["remaining_documents"].pop(matched_index)

        print(f"Evidence: {evidence}")
        print(f"Similarity score: {similarity}")

        if similarity < SIMILARITY_THRESHOLD:
            print("Verification: SKIPPED (Low similarity)")
            continue

        verification = verify_claim(claim_text, evidence)
        result = verification["status"]
        print(f"Verification: {result}")

        # Evidence gain measures improvement in supporting entailment confidence.
        current_support = (
            verification["confidence"]
            if "entailment" in verification["label"]
            else 0.0
        )
        previous_support = state["best_support_confidence"]
        evidence_gain = max(0.0, current_support - previous_support)

        state["best_support_confidence"] = max(
            previous_support, current_support
        )
        state["total_evidence_gain"] += evidence_gain

        # Penalize repeated low-gain retrievals.
        if evidence_gain < 0.05:
            state["low_gain_streak"] += 1
        else:
            state["low_gain_streak"] = 0

        print(f"Low evidence-gain streak: {state['low_gain_streak']}")

        # Update priority based on remaining uncertainty.
        remaining_uncertainty = max(
            0.0, 1.0 - state["best_support_confidence"]
        )
        state["dynamic_priority"] = (
            state["base_priority"] * remaining_uncertainty
        )
        print(
            f"Updated dynamic priority: "
            f"{state['dynamic_priority']:.4f}"
        )

        print(f"Evidence gain this retrieval: {evidence_gain:.4f}")
        print(
            f"Cumulative evidence gain: "
            f"{state['total_evidence_gain']:.4f}"
        )

        if result in ("SUPPORTED", "CONTRADICTED"):
            state["result"] = result
            state["resolved"] = True

            expected_label = (
                "entailment"
                if result == "SUPPORTED"
                else "contradiction"
            )

            citation = (
                find_source_sentence_citation(claim_text, documents, args.source)
                if result == "SUPPORTED"
                else None
            )
            if citation is None:
                citation = get_precise_citation(
                    claim_text,
                    evidence,
                    args.source,
                    expected_label=expected_label
                )

            if citation:
                claim_citations[claim_text] = citation
            else:
                claim_citations.pop(claim_text, None)
                print(
                    "No individual sentence passed citation verification."
                )

            print(f"Claim resolved: {result}")
        else:
            print("Claim remains unresolved; scheduler will reconsider it.")

    verification_results = {
        claim_text: state["result"]
        for claim_text, state in claim_states.items()
    }

    for claim_text, state in claim_states.items():
        print(f"\nFinal Result: {claim_text} -> {state['result']}")
        print(
            f"Total supporting evidence gain: "
            f"{state['total_evidence_gain']:.4f}"
        )

    # Claims not processed because the budget ran out remain insufficient
    for claim in selected:
        verification_results.setdefault(
            claim["claim"],
            "INSUFFICIENT"
        )

    print(
        f"\nTotal evidence retrievals used: "
        f"{retrievals_used}/{retrieval_budget}"
    )

    print("\n=== Answer Revision ===")

    revised_claims, flagged_claims, citation_missing_claims = revise_answer(
        claims,
        verification_results,
        claim_citations
    )

    print("\nFinal Answer:")
    for claim in revised_claims:
        citation = claim_citations.get(claim)

        if citation:
            print(
                f"- {claim} "
                f"[Source: {citation['source']}, Page {citation['page']}]"
            )
            print(f"  Supporting evidence: {citation['evidence']}")
            print(
                f"  Citation verification confidence: "
                f"{citation['confidence']:.4f}"
            )
        else:
            print(f"- {claim} [No retrieved citation available]")

    print("\nContradicted Claims and Evidence:")
    for claim in selected:
        if verification_results.get(claim["claim"]) == "CONTRADICTED":
            citation = claim_citations.get(claim["claim"])

            print(f"- {claim['claim']}")
            if citation:
                print(
                    f"  Source: {citation['source']}, "
                    f"Page {citation['page']}"
                )
                print(f"  Contradicting evidence: {citation['evidence']}")

    print("\nSupported Claims Missing a Verified Sentence-Level Citation:")
    for claim in citation_missing_claims:
        print(f"- {claim}")

    print("\nClaims Requiring Further Verification:")
    for claim in flagged_claims:
        print(f"- {claim}")
