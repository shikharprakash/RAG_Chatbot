from __future__ import annotations

import logging
import re
from typing import Protocol

from src.models import RankedChunk, RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RankedChunk]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RankedChunk]:
        if not chunks:
            return []
        try:
            scores = self._load_model().predict([(query, item.content) for item in chunks])
            ranked = [
                RankedChunk(**item.__dict__, rerank_score=float(score))
                for item, score in zip(chunks, scores, strict=True)
            ]
        except Exception as exc:
            logger.warning("Cross-encoder reranking unavailable; using lexical fallback: %s", exc)
            query_terms = set(re.findall(r"[a-z0-9_-]+", query.lower()))
            ranked = []
            for item in chunks:
                content_terms = set(re.findall(r"[a-z0-9_-]+", item.content.lower()))
                score = len(query_terms & content_terms) / max(len(query_terms), 1)
                ranked.append(RankedChunk(**item.__dict__, rerank_score=score))
        ranked.sort(key=lambda item: (item.rerank_score, item.retrieval_score), reverse=True)
        for rank, item in enumerate(ranked, start=1):
            item.final_rank = rank
        return ranked
