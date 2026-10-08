from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable

from src.config import settings

logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(
        self,
        ttl_seconds: int | None = None,
        similarity_threshold: float | None = None,
        file_path: Path | None = None,
        embed_query: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else settings.cache_ttl_seconds
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else settings.cache_similarity_threshold
        )
        self.file_path = file_path or settings.cache_file_path
        self._embed_query = embed_query
        self._backend = f"{settings.llm_base_url}:{settings.llm_model}"
        self._lock = threading.RLock()
        self._entries: list[dict[str, Any]] = []
        self._metrics = {"cache_hits": 0, "cache_misses": 0, "cache_hit_latency_ms": 0.0, "cache_miss_latency_ms": 0.0}
        self._load()

    def _embedding(self, query: str) -> list[float]:
        if self._embed_query:
            return self._embed_query(query)
        from src.vectorstore import get_embeddings

        return get_embeddings().embed_query(query)

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        norms = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
        return dot / norms if norms else 0.0

    def get(self, query: str) -> dict[str, Any] | None:
        started = time.perf_counter()
        try:
            embedding = self._embedding(query)
        except Exception as exc:
            logger.warning("Semantic cache embedding lookup failed; treating as miss: %s", exc)
            embedding = []
        now = time.time()
        with self._lock:
            self._entries = [entry for entry in self._entries if entry["expires_at"] > now]
            match = None
            best_score = -1.0
            for entry in self._entries:
                if entry.get("backend") != self._backend:
                    continue
                score = self._cosine(embedding, entry["query_embedding"]) if embedding else 0.0
                if score >= self.similarity_threshold and score > best_score:
                    match, best_score = entry, score
            elapsed = (time.perf_counter() - started) * 1000
            if match:
                self._metrics["cache_hits"] += 1
                self._metrics["cache_hit_latency_ms"] += elapsed
                self._persist()
                return dict(match["result"])
            self._metrics["cache_misses"] += 1
            self._metrics["cache_miss_latency_ms"] += elapsed
            self._persist()
            return None

    def set(self, query: str, result: dict[str, Any]) -> None:
        try:
            embedding = self._embedding(query)
        except Exception as exc:
            logger.warning("Could not embed query for semantic cache; response will not be cached: %s", exc)
            return
        now = time.time()
        with self._lock:
            self._entries.append({
                "query": query,
                "query_embedding": embedding,
                "result": result,
                "backend": self._backend,
                "created_at": now,
                "expires_at": now + self.ttl_seconds,
            })
            self._persist()

    def metrics(self) -> dict[str, float | int]:
        with self._lock:
            hits = int(self._metrics["cache_hits"])
            misses = int(self._metrics["cache_misses"])
            total = hits + misses
            return {
                **self._metrics,
                "cache_hit_rate": hits / total if total else 0.0,
                "active_entries": len(self._entries),
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._persist()

    def _load(self) -> None:
        try:
            if self.file_path.exists():
                data = json.loads(self.file_path.read_text(encoding="utf-8"))
                self._entries = [
                    item
                    for item in data
                    if item.get("expires_at", 0) > time.time()
                    and item.get("backend") == self._backend
                ]
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Semantic cache could not be loaded; starting empty: %s", exc)
            self._entries = []

    def _persist(self) -> None:
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.file_path.with_suffix(self.file_path.suffix + ".tmp")
            temporary.write_text(json.dumps(self._entries), encoding="utf-8")
            temporary.replace(self.file_path)
        except OSError as exc:
            logger.warning("Semantic cache persistence failed: %s", exc)
