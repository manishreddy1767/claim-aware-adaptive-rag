"""Fine-tune the NLI verifier on RAGTruth train (claim-level hallucination labels).

Examples: every claim extracted from a RAGTruth *train* response is paired with a
premise made of the top-k most relevant sentences of that response's source
(retrieved with the project's hybrid retriever, in document order). Labels
keep NLI semantics so the fine-tuned model is a drop-in replacement for the
verifier's NLI model:

    faithful claim            -> entailment
    'conflict' hallucination  -> contradiction
    'baseless' hallucination  -> neutral

10% of train *sources* are held out for validation; the RAGTruth test split is
never used here. Faithful claims are subsampled (``--neg-ratio``) to reduce
class imbalance.

Memory (4 GB GPU): fp16 autocast, gradient checkpointing, frozen word
embeddings (98M of deberta-v3-base's 184M parameters).

    python scripts/train_verifier.py build      # -> data/ragtruth/ft_{train,val}.jsonl
    python scripts/train_verifier.py train      # -> models/nli-deberta-v3-base-ragtruth/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from carag.claims import extract_claims  # noqa: E402
from carag.config import RAGConfig  # noqa: E402
from carag.ingestion import load_bytes  # noqa: E402
from carag.pipeline import ClaimAwareRAG  # noqa: E402
from evaluation.run_ragtruth import ensure_data, gold_type, sentence_spans, source_text  # noqa: E402

DATA = ROOT / "data" / "ragtruth"
OUT = ROOT / "models" / "nli-deberta-v3-base-ragtruth"
BASE = "cross-encoder/nli-deberta-v3-base"
LABEL_OF = {None: "entailment", "conflict": "contradiction", "baseless": "neutral"}


def claim_offsets(sentence: str, start: int, claim: str) -> tuple[int, int]:
    """Approximate character span of a claim inside its sentence (falls back to the sentence)."""
    squeeze = [(i, c.lower()) for i, c in enumerate(sentence) if c.isalnum()]
    text = "".join(c for _, c in squeeze)
    key = "".join(c.lower() for c in claim if c.isalnum())
    pos = text.find(key) if key else -1
    if pos < 0:
        return start, start + len(sentence)
    return start + squeeze[pos][0], start + squeeze[pos + len(key) - 1][0] + 1


def build(k: int, neg_ratio: float, seed: int) -> None:
    data = ensure_data()
    sources = {}
    for line in (data / "source_info.jsonl").read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        sources[s["source_id"]] = s
    responses = [json.loads(l) for l in (data / "response.jsonl").read_text(encoding="utf-8").splitlines()]
    train = [r for r in responses if r["split"] == "train"]
    rng = random.Random(seed)
    source_ids = sorted({r["source_id"] for r in train})
    rng.shuffle(source_ids)
    val_sources = set(source_ids[: len(source_ids) // 10])

    config = RAGConfig()
    config.answer.relevance_check = False
    by_source: dict[str, list] = {}
    for r in train:
        by_source.setdefault(r["source_id"], []).append(r)

    out = {"train": [], "val": []}
    t0 = time.perf_counter()
    for n, (sid, rs) in enumerate(by_source.items(), start=1):
        src = sources[sid]
        rag = ClaimAwareRAG(config)
        try:
            rag.add_document(load_bytes(source_text(src["task_type"], src["source_info"]).encode("utf-8"), "s.txt"))
        except Exception:
            continue
        split = "val" if sid in val_sources else "train"
        for r in rs:
            for sentence, start, end in sentence_spans(r["response"]):
                for claim in extract_claims(sentence):
                    a, b = claim_offsets(sentence, start, claim)
                    label = LABEL_OF[gold_type(r["labels"], a, b)]
                    if split == "train" and label == "entailment" and rng.random() > neg_ratio:
                        continue
                    top = rag.retriever.rank(claim, k, "hybrid")
                    premise = " ".join(e.unit.text for e in sorted(top, key=lambda e: e.unit.position))
                    out[split].append({"premise": premise, "hypothesis": claim, "label": label,
                                       "task": src["task_type"], "response_id": r["id"]})
        if n % 200 == 0:
            print(f"  {n}/{len(by_source)} sources, {len(out['train'])} train / {len(out['val'])} val "
                  f"examples ({time.perf_counter() - t0:.0f}s)", flush=True)
    for split, rows in out.items():
        path = DATA / f"ft_{split}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        counts = {l: sum(r["label"] == l for r in rows) for l in ("entailment", "contradiction", "neutral")}
        print(f"{split}: {len(rows)} examples {counts} -> {path}")


def evaluate(model, tokenizer, rows, label2id, device, max_len, batch_size=32) -> dict:
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            enc = tokenizer([r["premise"] for r in batch], [r["hypothesis"] for r in batch], truncation="longest_first",
                            max_length=max_len, padding=True, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**enc).logits
            preds += logits.argmax(-1).tolist()
    model.train()
    gold = [label2id[r["label"]] for r in rows]
    ent = label2id["entailment"]
    tp = sum(g != ent and p != ent for g, p in zip(gold, preds))
    fp = sum(g == ent and p != ent for g, p in zip(gold, preds))
    fn = sum(g != ent and p == ent for g, p in zip(gold, preds))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy_3way": round(sum(g == p for g, p in zip(gold, preds)) / len(gold), 4),
            "halluc_precision": round(precision, 4), "halluc_recall": round(recall, 4),
            "halluc_f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0}


def train(epochs: float, batch_size: int, accum: int, lr: float, max_len: int, seed: int, limit: int | None) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [json.loads(l) for l in (DATA / "ft_train.jsonl").read_text(encoding="utf-8").splitlines()]
    val = [json.loads(l) for l in (DATA / "ft_val.jsonl").read_text(encoding="utf-8").splitlines()]
    if limit:
        rows = rows[:limit]
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE).to(device)
    label2id = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    model.gradient_checkpointing_enable()
    for p in model.deberta.embeddings.word_embeddings.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    steps = int(len(rows) * epochs / (batch_size * accum))
    steps_per_epoch = max(1, len(rows) // (batch_size * accum))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    print(f"train {len(rows)} val {len(val)} | trainable params {sum(p.numel() for p in trainable) / 1e6:.0f}M "
          f"| {steps} optimizer steps", flush=True)
    print("before fine-tuning:", evaluate(model, tokenizer, val, label2id, device, max_len), flush=True)
    model.train()
    order = []
    step, micro, t0 = 0, 0, time.perf_counter()
    while step < steps:
        if not order:
            order = list(range(len(rows)))
            random.shuffle(order)
        idx = [order.pop() for _ in range(min(batch_size, len(order)))]
        batch = [rows[i] for i in idx]
        enc = tokenizer([r["premise"] for r in batch], [r["hypothesis"] for r in batch], truncation="longest_first",
                        max_length=max_len, padding=True, return_tensors="pt").to(device)
        labels = torch.tensor([label2id[r["label"]] for r in batch], device=device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            loss = model(**enc, labels=labels).loss / accum
        scaler.scale(loss).backward()
        micro += 1
        if micro % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if step % steps_per_epoch == 0 and step < steps:
                epoch_metrics = evaluate(model, tokenizer, val, label2id, device, max_len)
                print(f"  epoch checkpoint at step {step}: {epoch_metrics}", flush=True)
                OUT.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(OUT)
                tokenizer.save_pretrained(OUT)
            if step % 100 == 0:
                print(f"  step {step}/{steps} loss {loss.item() * accum:.3f} "
                      f"({time.perf_counter() - t0:.0f}s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB)",
                      flush=True)
    metrics = evaluate(model, tokenizer, val, label2id, device, max_len)
    print("after fine-tuning:", metrics, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
    (OUT / "training_info.json").write_text(json.dumps({
        "base_model": BASE, "train_examples": len(rows), "val_examples": len(val), "epochs": epochs,
        "batch_size": batch_size, "grad_accum": accum, "lr": lr, "max_len": max_len, "seed": seed,
        "val_metrics": metrics, "train_time_s": round(time.perf_counter() - t0, 1)}, indent=2), encoding="utf-8")
    print(f"saved to {OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--k", type=int, default=5, help="premise = top-k retrieved source sentences")
    b.add_argument("--neg-ratio", type=float, default=0.35, help="fraction of faithful train claims kept")
    b.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--accum", type=int, default=2)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--max-len", type=int, default=320)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--limit", type=int, default=None, help="use only the first N examples (smoke test)")
    args = parser.parse_args()
    if args.cmd == "build":
        build(args.k, args.neg_ratio, args.seed)
    else:
        train(args.epochs, args.batch_size, args.accum, args.lr, args.max_len, args.seed, args.limit)


if __name__ == "__main__":
    main()
