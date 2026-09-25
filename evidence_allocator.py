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


# Step 2: Allocate retrieval budget
def allocate_budget(claims, budget):
    candidates = [
        claim for claim in claims
        if claim["status"] != "Supported"
    ]

    return sorted(
        candidates,
        key=lambda claim: claim["evidence_gap"],
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

    # Resolve generic labels using model configuration
    if label.startswith("label_"):
        label_id = int(label.split("_")[1])
        id2label = nli_model.model.config.id2label
        label = id2label.get(
            label_id,
            label
        ).lower()

    print(f"NLI Label: {label}")
    print(f"NLI Confidence: {confidence:.4f}")

    if confidence < 0.70:
        return "INSUFFICIENT"

    if "entailment" in label:
        return "SUPPORTED"

    if "contradiction" in label:
        return "CONTRADICTED"

    return "INSUFFICIENT"


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

    print("\n=== Evidence Gap Analysis ===")
    for claim in claims:
        print(
            f'{claim["claim"]}: {claim["status"]}, '
            f'Gap = {claim["evidence_gap"]}'
        )

    selected = allocate_budget(claims, budget)

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

    # Shared budget: maximum number of evidence passages to retrieve
    retrieval_budget = 6
    retrievals_used = 0

    for claim in selected:
        claim_text = claim["claim"]
        print(f"\nClaim: {claim_text}")

        final_result = "INSUFFICIENT"
        remaining_documents = documents.copy()
        round_number = 0

        while (
            remaining_documents
            and retrievals_used < retrieval_budget
        ):
            round_number += 1
            retrievals_used += 1

            print(f"\n--- Retrieval Round {round_number} ---")
            print(
                f"Shared retrieval budget remaining: "
                f"{retrieval_budget - retrievals_used}"
            )

            # Dynamically retrieve the best remaining passage
            evidence_list = retrieve_evidence(
                claim_text,
                remaining_documents,
                top_k=1
            )

            if not evidence_list:
                print("No more evidence available.")
                break

            item = evidence_list[0]
            evidence = item["document"]
            similarity = item["score"]

            # Remove this passage so it cannot be retrieved again
            remaining_documents.remove(evidence)

            print(f"Evidence: {evidence}")
            print(f"Similarity score: {similarity}")

            if similarity < SIMILARITY_THRESHOLD:
                print("Verification: SKIPPED (Low similarity)")
                continue

            result = verify_claim(claim_text, evidence)
            print(f"Verification: {result}")

            if result in ("SUPPORTED", "CONTRADICTED"):
                final_result = result

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

                print(f"Stopping retrieval: {final_result}")
                break

            print("Evidence insufficient. Retrieving another passage.")

        verification_results[claim_text] = final_result
        print(f"Final Result: {final_result}")

        if retrievals_used >= retrieval_budget:
            print("\nShared retrieval budget exhausted.")
            break

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
