"""
    Two-level response cache backed by Redis.

    Level 1: exact-match cache using a SHA-256 hash of the normalized
    question text.

    Level 2: semantic cache using embedding similarity above a configurable
    threshold.

    Both levels share the same Redis instance and are checked in order:
    exact first, since it is cheaper, followed by semantic matching.

    The TTL is configurable and defaults to 24 hours.

    Cache entries can be invalidated by calling ``invalidate_all()`` after
    document ingestion.
"""

import hashlib
import numpy as np
import time
import json
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
        self._ttl = ttl_seconds
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

    def _sem_index_key(self) -> str:
        return "cache:sem:index"

    def get_semantic(self, question: str) -> Optional[str]:

        entries_raw = self._r.lrange(self._sem_index_key(), 0, -1)

        if not entries_raw:
            return None

        query_vec = np.array(
            _get_embeddings().embed_query(question), dtype=np.float32
        )

        best_score = -1.0
        best_answer: Optional[str] = None

        for raw in entries_raw:
            entry = json.loads(raw)
            cached_vec = np.array(entry["vector"], dtype=np.float32)
            score = float(np.dot(query_vec, cached_vec))

            if score > best_score:
                best_score = score
                best_answer = entry["answer"]

        if best_score >= self._sem_threshold and best_answer:
            log.info("cache.semantic_hit",
                     score=round(best_score, 3),
                     question_preview=question[:60])
            return best_answer

    def set_semantic(self, question: str, answer: str) -> None:

        vec = _get_embeddings().embed_query(question)

        entry = json.dumps(
            {"vector": vec,
             "answer": answer,
             "question": question,
             "ts": time.time()})
        self._r.rpush(self._sem_index_key(), entry)
        self._r.expire(self._sem_index_key(), self._ttl)

    def get(self, question: str) -> Optional[str]:
        return self.get_exact(question) or self.get_semantic(question)

    def set(self, question: str, answer: str) -> None:

        self.set_exact(question, answer)
        self.set_semantic(question, answer)

    def invalidate_all(self) -> None:

        keys = self._r.keys("cache:*")
        if keys:
            self._r.delete(*keys)
        log.info("cache.invalidated", deleted=len(keys))
