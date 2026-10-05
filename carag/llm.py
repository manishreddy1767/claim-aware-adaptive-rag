"""Local LLMs served by Ollama (https://ollama.com), e.g. qwen3:8b.

The model only drafts the answer from the retrieved evidence; the draft is
then verified claim by claim against the documents (pipeline.ask_with_llm),
so unsupported statements are removed or corrected before they are shown.
Nothing leaves the computer: Ollama runs locally.
"""

from __future__ import annotations

import os
import re

import requests

from .schema import ScoredEvidence

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:8b"
NO_ANSWER = "The documents do not contain this information."

_SYSTEM = f"""You answer questions using only the numbered passages from the user's documents.
Rules:
- Use only facts stated in the passages. You may combine passages and reason with them (compare numbers, \
apply a rule to the case in the question), but never add outside knowledge.
- Write short, complete sentences, one fact per sentence, each one understandable on its own.
- For a yes/no question, start with "Yes." or "No." and then give the facts.
- If the passages disagree, say so and give both statements.
- Do not mention passage numbers.
- If the passages do not contain the answer, reply exactly: {NO_ANSWER}"""

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


class OllamaError(RuntimeError):
    """Ollama is not running, the model is missing, or generation failed."""


class OllamaClient:
    def __init__(self, host: str | None = None, timeout_s: float = 600):
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        if not self.host.startswith(("http://", "https://")):
            self.host = "http://" + self.host
        self.timeout_s = timeout_s

    def models(self) -> list[str]:
        """Installed model names; raises OllamaError when Ollama is not reachable."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=3)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError("Ollama is not running. Install it from https://ollama.com and start it.") from exc
        return sorted(m["name"] for m in response.json().get("models", []))

    def status(self, model: str) -> dict:
        try:
            installed = self.models()
        except OllamaError as exc:
            return {"running": False, "installed": [], "model_ready": False, "message": str(exc)}
        ready = model in installed or f"{model}:latest" in installed
        message = "Ready." if ready else f"The model '{model}' is not installed. Run: ollama pull {model}"
        return {"running": True, "installed": installed, "model_ready": ready, "message": message}

    def chat(self, model: str, system: str, user: str, max_tokens: int = 512) -> str:
        payload = {"model": model, "stream": False, "think": False, "keep_alive": "15m",
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "options": {"temperature": 0, "num_ctx": 8192, "num_predict": max_tokens}}
        try:
            response = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise OllamaError(f"Could not reach Ollama at {self.host}: {exc}") from exc
        if response.status_code == 404:
            raise OllamaError(f"The model '{model}' is not installed. Run: ollama pull {model}")
        if not response.ok:
            raise OllamaError(f"Ollama could not answer ({response.status_code}): {response.text[:200]}")
        text = response.json().get("message", {}).get("content", "")
        return _THINK.sub("", text).strip()


class OllamaGenerator:
    """Drafts an answer from numbered evidence passages with an Ollama model."""

    def __init__(self, model: str = DEFAULT_MODEL, client: OllamaClient | None = None):
        self.model = model
        self.client = client or OllamaClient()

    def generate(self, question: str, evidence: list[ScoredEvidence]) -> str:
        passages = []
        for i, e in enumerate(evidence, start=1):
            where = e.unit.source + (f", {e.unit.section}" if e.unit.section else "")
            passages.append(f"[{i}] ({where}) {e.unit.text}")
        user = "Passages:\n" + "\n".join(passages) + f"\n\nQuestion: {question}"
        return self.client.chat(self.model, _SYSTEM, user)
