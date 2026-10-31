"""External fact-checking / hallucination-detection baselines.

* MiniCheck (Tang et al., EMNLP 2024): lytang/MiniCheck-DeBERTa-v3-Large, a
  2-class (supported / unsupported) cross-encoder. Following the authors' usage,
  the document is split into chunks of about 400 tokens and the claim's support
  score is the maximum over chunks.
* HHEM-2.1-Open (Vectara): vectara/hallucination_evaluation_model, a
  Flan-T5-based consistency scorer in [0, 1]; scored per chunk, max over chunks.

Both output P(supported); a claim is predicted unsupported when it is < 0.5.
They are used for evaluation only, not in the application.
"""

from __future__ import annotations

import numpy as np

from carag.models import _load, resolve_device
from carag.text_utils import split_sentences


def chunk_document(document: str, tokenizer, max_tokens: int = 400) -> list[str]:
    """Greedy sentence packing into chunks of at most ``max_tokens`` tokens."""
    chunks, current, length = [], [], 0
    for sentence in split_sentences(document) or [document]:
        n = len(tokenizer.tokenize(sentence))
        if current and length + n > max_tokens:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(sentence)
        length += n
    if current:
        chunks.append(" ".join(current))
    return chunks or [document]


class MiniCheck:
    name = "minicheck_deberta_large"

    def __init__(self, model_name: str = "lytang/MiniCheck-DeBERTa-v3-Large", device: str = "auto",
                 batch_size: int = 8):
        self.batch_size = batch_size

        def factory(dev):
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_name)
            dtype = torch.float16 if dev == "cuda" else torch.float32
            model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=dtype).to(dev).eval()
            return tok, model

        (self.tokenizer, self.model), self.device = _load("minicheck", model_name, resolve_device(device), factory)
        self._chunks: dict[str, list[str]] = {}

    def score(self, document: str, claims: list[str]) -> list[float]:
        import torch
        if document not in self._chunks:
            self._chunks = {document: chunk_document(document, self.tokenizer)}
        chunks = self._chunks[document]
        pairs = [(c, claim) for claim in claims for c in chunks]
        probs = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            enc = self.tokenizer([p for p, _ in batch], [h for _, h in batch], truncation="only_first",
                                 max_length=512, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits.float()
            probs += torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
        probs = np.array(probs).reshape(len(claims), len(chunks))
        return probs.max(axis=1).tolist()


class HHEM:
    name = "hhem_2_1_open"
    PROMPT = ("<pad> Determine if the hypothesis is true given the premise?\n\n"
              "Premise: {text1}\n\nHypothesis: {text2}")

    def __init__(self, model_name: str = "vectara/hallucination_evaluation_model", device: str = "auto",
                 batch_size: int = 8):
        self.batch_size = batch_size

        def factory(dev):
            # The published remote code is incompatible with transformers 5, so the same
            # architecture (Flan-T5-base token classifier, 2 labels) is rebuilt from
            # standard classes and the released weights are loaded directly.
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            from transformers import AutoConfig, AutoTokenizer, T5ForTokenClassification
            config = AutoConfig.from_pretrained("google/flan-t5-base", num_labels=2)
            model = T5ForTokenClassification(config)
            state = load_file(hf_hub_download(model_name, "model.safetensors"))
            state = {k.removeprefix("t5."): v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            missing = [k for k in missing if "embed_tokens" not in k]   # tied to shared embeddings
            if missing or unexpected:
                raise RuntimeError(f"HHEM weights mismatch: missing={missing}, unexpected={unexpected}")
            return AutoTokenizer.from_pretrained("google/flan-t5-base"), model.to(dev).eval()

        (self.tokenizer, self.model), self.device = _load("hhem", model_name, resolve_device(device), factory)
        self._chunks: dict[str, list[str]] = {}

    def score(self, document: str, claims: list[str]) -> list[float]:
        import torch
        if document not in self._chunks:
            self._chunks = {document: chunk_document(document, self.tokenizer)}
        chunks = self._chunks[document]
        pairs = [(c, claim) for claim in claims for c in chunks]
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            prompts = [self.PROMPT.format(text1=p, text2=h) for p, h in pairs[i:i + self.batch_size]]
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                                 max_length=2048).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, 0, :].float()   # first-token classification
            scores += torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
        return np.array(scores).reshape(len(claims), len(chunks)).max(axis=1).tolist()
