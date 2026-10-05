"""Local LLMs: find whichever one runs on this computer and use it to draft answers.

Supported: Ollama (its own API) and every server with the OpenAI-compatible chat API,
which covers LM Studio, Jan, GPT4All, llama.cpp's server, LocalAI, vLLM, KoboldCpp and
text-generation-webui. ``discover`` probes their usual local ports (all at once, with a
short timeout) and lists the chat models each one offers; ``choose`` picks one.

The model only drafts the answer from the retrieved evidence; the draft is then verified
claim by claim against the documents (pipeline.ask_with_llm), so unsupported statements
are removed or corrected before they are shown. Nothing leaves the computer.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import requests

from .schema import ScoredEvidence

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:8b"
NO_ANSWER = "The documents do not contain this information."

# Where local LLM servers listen by default: (name, URL, API kind).
KNOWN_SERVERS = [
    ("Ollama", DEFAULT_HOST, "ollama"),
    ("LM Studio", "http://127.0.0.1:1234", "openai"),
    ("Jan", "http://127.0.0.1:1337", "openai"),
    ("GPT4All", "http://127.0.0.1:4891", "openai"),
    ("llama.cpp / LocalAI", "http://127.0.0.1:8080", "openai"),
    ("vLLM", "http://127.0.0.1:8000", "openai"),
    ("KoboldCpp", "http://127.0.0.1:5001", "openai"),
    ("text-generation-webui", "http://127.0.0.1:5000", "openai"),
]

# Preferred chat models, best first (matched as prefixes of the installed names).
PREFERRED = ["qwen3", "qwen2.5", "llama3.3", "llama3.1", "llama3.2", "gemma3", "mistral", "phi4", "gemma2",
             "phi3", "deepseek-r1", "llama3", "qwen2", "gemma"]
_NOT_CHAT = re.compile(r"embed|bge-|minilm|rerank|clip|whisper|tts|e5-|nomic-embed", re.I)

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


class LLMError(RuntimeError):
    """The model server is not running, the model is missing, or generation failed."""


OllamaError = LLMError   # earlier name


def _normalize(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url[:-3] if url.endswith("/v1") else url


class OllamaClient:
    kind = "ollama"

    def __init__(self, host: str | None = None, timeout_s: float = 600):
        self.host = _normalize(host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST)
        self.timeout_s = timeout_s

    def models(self, timeout_s: float = 3) -> list[str]:
        """Installed model names; raises LLMError when Ollama is not reachable."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=timeout_s)
            response.raise_for_status()
            return sorted(m["name"] for m in response.json().get("models", []))
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            raise LLMError("Ollama is not running. Install it from https://ollama.com and start it.") from exc

    def status(self, model: str) -> dict:
        try:
            installed = self.models()
        except LLMError as exc:
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
            raise LLMError(f"Could not reach Ollama at {self.host}: {exc}") from exc
        if response.status_code == 404:
            raise LLMError(f"The model '{model}' is not installed. Run: ollama pull {model}")
        if not response.ok:
            raise LLMError(f"Ollama could not answer ({response.status_code}): {response.text[:200]}")
        try:
            text = response.json().get("message", {}).get("content", "")
        except ValueError as exc:
            raise LLMError("Ollama returned an unreadable reply.") from exc
        return _THINK.sub("", text).strip()


class OpenAICompatibleClient:
    """LM Studio, Jan, GPT4All, llama.cpp, LocalAI, vLLM, KoboldCpp, text-generation-webui..."""

    kind = "openai"

    def __init__(self, host: str, timeout_s: float = 600, name: str = "Local server"):
        self.host = _normalize(host)
        self.timeout_s = timeout_s
        self.name = name

    def models(self, timeout_s: float = 3) -> list[str]:
        try:
            response = requests.get(f"{self.host}/v1/models", timeout=timeout_s)
            response.raise_for_status()
            data = response.json()["data"]
            return sorted(str(m["id"]) for m in data)
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            raise LLMError(f"No OpenAI-compatible server is answering at {self.host}.") from exc

    def chat(self, model: str, system: str, user: str, max_tokens: int = 512) -> str:
        payload = {"model": model, "stream": False, "temperature": 0, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        try:
            response = requests.post(f"{self.host}/v1/chat/completions", json=payload, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach {self.name} at {self.host}: {exc}") from exc
        if not response.ok:
            raise LLMError(f"{self.name} could not answer ({response.status_code}): {response.text[:200]}")
        try:
            text = response.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.name} returned an unreadable reply.") from exc
        return _THINK.sub("", text).strip()


def client_for(name: str, url: str, kind: str):
    return OllamaClient(url) if kind == "ollama" else OpenAICompatibleClient(url, name=name)


@dataclass
class Server:
    """A local LLM server found on this computer, with its chat models."""
    name: str
    url: str
    kind: str
    models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "url": self.url, "kind": self.kind, "models": self.models}


def known_servers(extra_url: str | None = None) -> list[tuple[str, str, str]]:
    """The usual local servers, with OLLAMA_HOST honoured and an optional extra (custom) URL."""
    servers = list(KNOWN_SERVERS)
    if os.environ.get("OLLAMA_HOST"):
        servers[0] = ("Ollama", _normalize(os.environ["OLLAMA_HOST"]), "ollama")
    if extra_url:
        servers.insert(0, ("Custom server", _normalize(extra_url), "openai"))
    return servers


def discover(candidates: list[tuple[str, str, str]] | None = None, timeout_s: float = 0.8) -> list[Server]:
    """Every candidate server that answers, with its chat models (embedding models are left out).
    All candidates are probed at once, so this takes at most about ``timeout_s``."""
    candidates = candidates if candidates is not None else known_servers()

    def probe(candidate):
        name, url, kind = candidate
        try:
            models = client_for(name, url, kind).models(timeout_s=timeout_s)
        except LLMError:
            return None
        chat_models = [m for m in models if not _NOT_CHAT.search(m)]
        return Server(name, _normalize(url), kind, chat_models) if chat_models else None

    with ThreadPoolExecutor(max_workers=max(1, len(candidates))) as pool:
        found = list(pool.map(probe, candidates))
    seen, servers = set(), []
    for server in found:
        if server is not None and server.url not in seen:   # one entry per URL
            seen.add(server.url)
            servers.append(server)
    return servers


def _rank(model: str) -> int:
    low = model.lower()
    return next((i for i, prefix in enumerate(PREFERRED) if low.startswith(prefix)), len(PREFERRED))


def choose(servers: list[Server], model: str = "auto", server_url: str | None = None) -> tuple[Server, str] | None:
    """The (server, model) to use: the requested model where it is installed, otherwise the
    best-known chat model on any server (Ollama first). None when nothing is available."""
    if server_url:
        servers = [s for s in servers if s.url == _normalize(server_url)] or servers
    if model and model != "auto":
        for server in servers:
            for name in server.models:
                if name == model or name == f"{model}:latest":
                    return server, name
    best = None
    for order, server in enumerate(servers):
        for name in server.models:
            key = (_rank(name), order)
            if best is None or key < best[0]:
                best = (key, server, name)
    return (best[1], best[2]) if best else None


class LLMGenerator:
    """Drafts an answer from numbered evidence passages with any supported local model."""

    def __init__(self, model: str = DEFAULT_MODEL, client=None, provider: str | None = None):
        self.model = model
        self.client = client or OllamaClient()
        self.provider = provider or ("Ollama" if self.client.kind == "ollama" else getattr(self.client, "name", ""))

    def generate(self, question: str, evidence: list[ScoredEvidence]) -> str:
        passages = []
        for i, e in enumerate(evidence, start=1):
            where = e.unit.source + (f", {e.unit.section}" if e.unit.section else "")
            passages.append(f"[{i}] ({where}) {e.unit.text}")
        user = "Passages:\n" + "\n".join(passages) + f"\n\nQuestion: {question}"
        return self.client.chat(self.model, _SYSTEM, user)


OllamaGenerator = LLMGenerator   # earlier name
