"""Download and load every model once (run by install.bat), so the application
starts quickly and then works without an internet connection.

    python -m carag.server.prefetch
"""

from __future__ import annotations

import logging
import sys
import time


def quiet_model_libraries() -> None:
    """Hide download/loading chatter from the model libraries (not useful to end users)."""
    import os
    import warnings
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
    for name in ("huggingface_hub", "transformers", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    quiet_model_libraries()
    from ..config import RAGConfig
    from ..models import ModelLoadError, resolve_device
    from ..pipeline import ClaimAwareRAG

    config = RAGConfig()
    device = resolve_device(config.models.device)
    print(f"Preparing the models on {'the GPU' if device == 'cuda' else 'the CPU'} "
          "(the first time this downloads about 1 GB)...")
    start = time.perf_counter()
    try:
        rag = ClaimAwareRAG(config)
        print(f"  embedding model: {config.models.embedding_model}")
        rag.answerer
        print(f"  question-answering model: {config.answer.relevance_model}")
        rag.verifier
        print(f"  claim-checking model: {config.models.nli_model}")
    except ModelLoadError as exc:
        print(f"Could not prepare the models: {exc}", file=sys.stderr)
        return 1
    print(f"Models ready in {time.perf_counter() - start:.0f} s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
