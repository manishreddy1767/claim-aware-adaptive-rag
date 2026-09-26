"""Optional local generative answering, filtered by claim verification.

A small instruction-tuned LLM drafts an answer from the retrieved evidence;
the draft then goes through ``check_answer`` (claim extraction, budgeted NLI
verification, revision), so unsupported statements are removed or corrected
before they reach the user. Off by default: the extractive answerer needs no
LLM. Default model: Qwen/Qwen2.5-0.5B-Instruct (~1 GB, fits a 4 GB GPU next to
the other models).
"""

from __future__ import annotations

from .models import _load, resolve_device
from .schema import ScoredEvidence

_SYSTEM = ("You answer questions using only the numbered evidence provided. "
           "Answer in one or two short sentences. If the evidence does not contain the answer, "
           "reply exactly: The evidence does not contain the answer.")

NO_ANSWER_MARKERS = ("does not contain the answer", "not contain the answer", "does not mention",
                     "no information", "not provided", "cannot be determined", "not specified",
                     "not stated", "does not say", "is not mentioned")


class LocalGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "auto",
                 max_new_tokens: int = 96):
        self.max_new_tokens = max_new_tokens

        def factory(dev):
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            dtype = torch.float16 if dev == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(dev).eval()
            return tokenizer, model

        (self.tokenizer, self.model), self.device = _load("llm", model_name, resolve_device(device), factory)

    def generate(self, question: str, evidence: list[ScoredEvidence]) -> str:
        import torch

        numbered = "\n".join(f"[{i}] {e.unit.text}" for i, e in enumerate(evidence, start=1))
        messages = [{"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": f"Evidence:\n{numbered}\n\nQuestion: {question}"}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()


def is_no_answer(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in NO_ANSWER_MARKERS)
