import hashlib
import redis
import structlog
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from rag_chatbot.config import get_settings


log = structlog.get_logger(__name__)

# Loaded once at module import; shared across all requests in this process.
_EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        cfg = get_settings()
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=cfg.embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDINGS


class ResponseCache:

    def __init__(self, ttl_seconds: int = 86400) -> None:
        cfg = get_settings()
        self._r = redis.from_url(cfg.redis_url, decode_responses=True)
        self._ttl = ttl_seconds,
        self._sem_threshold = 0.92

    def _exact_key(self, question: str) -> str:
        normalized = question.lower().strip()
        return "cache:exact:" + hashlib.sha256(normalized.encode()).hexdigest()

    def get_exact(self, question: str) -> Optional[str]:
        value = self._r.get(self._exact_key(question))
        if value:
            log.info("cache.exact_hit", question_preview=question[:60])
        return value

    def set_exact(self, question: str, answer: str) -> None:
        self._r.setex(self._exact_key(question), self._ttl, answer)
