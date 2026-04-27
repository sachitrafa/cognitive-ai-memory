import threading

from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None
_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = SentenceTransformer("all-mpnet-base-v2")
    return _model


def embed(text: str) -> list[float]:
    return _get_model().encode(text).tolist()
