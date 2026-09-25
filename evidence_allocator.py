import re
import argparse
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
    candidates = [
        claim for claim in claims
        if claim["status"] != "Supported"
    ]

    for claim in candidates:
        score = claim["support_score"]

        # Uncertainty is highest when the support score is near 0.5.
        uncertainty = 1 - abs(score - 0.5) * 2

        # Combine evidence gap and uncertainty into a priority score.
        claim["uncertainty"] = round(uncertainty, 2)
        claim["adaptive_priority"] = round(
            0.7 * claim["evidence_gap"] + 0.3 * uncertainty,
            3
        )

    return sorted(
        candidates,
        key=lambda claim: claim["adaptive_priority"],
        reverse=True
    )[:budget]


# Step 3: Semantic evidence retrieval
def retrieve_evidence(claim, documents, top_k=2):
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



# Extract the most relevant sentence for a precise citation
def get_precise_citation(
    claim, evidence, source, expected_label="entailment"
):
    page_match = re.search(r"\[Page (\d+)\]", evidence)
    page_number = page_match.group(1) if page_match else "Unknown"

    text = re.sub(r"\[Page \d+\]\s*", "", evidence)
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
def revise_answer(claims, verification_results):
    revised_claims = []
    flagged_claims = []

    for claim in claims:
        claim_text = claim["claim"]

        if claim["status"] == "Supported":
            revised_claims.append(claim_text)
            continue

        result = verification_results.get(
            claim_text,
            "INSUFFICIENT"
        )

        if result == "SUPPORTED":
            revised_claims.append(claim_text)

        elif result == "CONTRADICTED":
            print(f"Removed contradicted claim: {claim_text}")

        else:
            flagged_claims.append(claim_text)
            print(
                f"Flagged for insufficient evidence: {claim_text}"
            )

    return revised_claims, flagged_claims


# Sample claims
claims = [
    {
        "claim": "The Earth revolves around the Sun",
        "support_score": 0.9
    },
    {
        "claim": "The Moon revolves around Mars",
        "support_score": 0.2
    },
    {
        "claim": "The Sun is a star",
        "support_score": 0.6
    },
    {
        "claim": "Jupiter has rings",
        "support_score": 0.3
    },
]


# PDF documents are loaded from the command line inside main.

# Retrieval budget
budget = 3


# Main execution
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Claim-Aware Adaptive RAG with PDF ingestion"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF document to ingest"
    )
    args = parser.parse_args()

    documents = load_pdf_chunks(args.pdf_path)


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
                    args.pdf_path,
                    expected_label="entailment"
                )

    # Shared budget with adaptive claim scheduling
    retrieval_budget = 6
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
            "resolved": False,
        }
        for claim in selected
    }

    while retrievals_used < retrieval_budget:
        active = [
            state for state in claim_states.values()
            if not state["resolved"] and state["remaining_documents"]
        ]

        if not active:
            break

        # Favor high-priority claims while distributing retrieval attempts
        # across unresolved claims.
        state = max(
            active,
            key=lambda item: (
                item["dynamic_priority"]
                / (1 + 0.5 * item["attempts"])
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

        state["remaining_documents"].remove(evidence)

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

            citation = get_precise_citation(
                claim_text,
                evidence,
                args.pdf_path,
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

    revised_claims, flagged_claims = revise_answer(
        claims,
        verification_results
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

    print("\nClaims Requiring Further Verification:")
    for claim in flagged_claims:
        print(f"- {claim}")
