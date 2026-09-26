"""Answer-relevance checking with an extractive QA model (optional).

Verification only checks that an answer sentence is *supported by* the sources;
it cannot tell whether the sentence *answers the question*. An extractive QA
model trained with unanswerable questions (SQuAD 2.0) is used to:

* locate the answer span inside the retrieved evidence, so the sentence
  containing it becomes the answer sentence, and
* estimate answerability: the margin between the best span score and the
  "no answer" score. A negative margin means the model prefers "no answer".

Model: deepset/minilm-uncased-squad2 by default (~130 MB). Note that it was
trained on SQuAD 2.0 train, so results on SQuAD 2.0 dev are in-distribution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .models import _load, resolve_device

logger = logging.getLogger(__name__)


@dataclass
class SpanAnswer:
    text: str
    start: int          # character offsets in the context
    end: int
    margin: float       # best span logit sum - null logit sum (>0: model prefers an answer)


class SpanQA:
    def __init__(self, model_name: str = "deepset/minilm-uncased-squad2", device: str = "auto",
                 max_length: int = 384, stride: int = 128, max_answer_tokens: int = 30):
        self.max_length = max_length
        self.stride = stride
        self.max_answer_tokens = max_answer_tokens

        def factory(dev):
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(dev).eval()
            return tokenizer, model

        (self.tokenizer, self.model), self.device = _load("qa", model_name, resolve_device(device), factory)

    def answer(self, question: str, context: str) -> SpanAnswer:
        """Best span over all context windows, with its margin over 'no answer'."""
        import torch

        enc = self.tokenizer(question, context, truncation="only_second", max_length=self.max_length,
                             stride=self.stride, return_overflowing_tokens=True,
                             return_offsets_mapping=True, padding=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping").tolist()
        enc.pop("overflow_to_sample_mapping", None)
        with torch.no_grad():
            out = self.model(**{k: v.to(self.device) for k, v in enc.items()})
        starts = out.start_logits.float().cpu().numpy()
        ends = out.end_logits.float().cpu().numpy()

        best = SpanAnswer("", 0, 0, -np.inf)
        null_score = np.inf
        for w in range(starts.shape[0]):
            seq_ids = enc.sequence_ids(w)
            ctx = np.array([s == 1 for s in seq_ids])
            null_score = min(null_score, starts[w, 0] + ends[w, 0])
            s_logits = np.where(ctx, starts[w], -np.inf)
            e_logits = np.where(ctx, ends[w], -np.inf)
            top_s = np.argsort(-s_logits)[:20]
            top_e = np.argsort(-e_logits)[:20]
            for s in top_s:
                for e in top_e:
                    if e < s or e - s + 1 > self.max_answer_tokens or not (ctx[s] and ctx[e]):
                        continue
                    score = s_logits[s] + e_logits[e]
                    if score > best.margin:
                        start, end = offsets[w][s][0], offsets[w][e][1]
                        best = SpanAnswer(context[start:end], start, end, float(score))
        if best.margin == -np.inf:
            return SpanAnswer("", 0, 0, -np.inf)
        best.margin = float(best.margin - null_score)
        return best
