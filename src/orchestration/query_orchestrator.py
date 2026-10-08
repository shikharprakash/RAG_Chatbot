from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict

from src.cache.semantic_cache import SemanticCache
from src.config import settings
from src.generation.answer_generator import AnswerGenerator
from src.models import QueryResult
from src.retrieval.query_expander import QueryExpander
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import MultiQueryRetriever
from src.routing.classifier import QueryClassifier

logger = logging.getLogger(__name__)


class QueryOrchestrator:
    def __init__(
        self,
        cache: SemanticCache | None = None,
        classifier: QueryClassifier | None = None,
        expander: QueryExpander | None = None,
        retriever: MultiQueryRetriever | None = None,
        reranker=None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        self.cache = cache or SemanticCache()
        self.classifier = classifier or QueryClassifier()
        self.expander = expander or QueryExpander()
        self.retriever = retriever or MultiQueryRetriever()
        self.reranker = reranker or CrossEncoderReranker(settings.reranker_model)
        self.generator = generator or AnswerGenerator()

    def answer(self, query: str) -> QueryResult:
        query = query.strip()
        if not query:
            raise ValueError("Please enter a question.")
        request_id = str(uuid.uuid4())[:8]
        started = time.perf_counter()
        stages: dict[str, float] = {}
        cache_started = time.perf_counter()
        try:
            cached = self.cache.get(query)
        except Exception as exc:
            logger.warning("Cache lookup failed for request %s: %s", request_id, exc)
            cached = None
        stages["cache_lookup_ms"] = (time.perf_counter() - cache_started) * 1000
        if cached:
            cached["query"] = query
            cached["cache_hit"] = True
            cached["latency_ms"] = (time.perf_counter() - started) * 1000
            cached.setdefault("stage_timings_ms", {})
            cached["stage_timings_ms"].update(stages)
            result = QueryResult.from_dict(cached)
            logger.info("request_id=%s cache=hit latency_ms=%.1f", request_id, result.latency_ms)
            return result

        stage_start = time.perf_counter()
        classification = self.classifier.classify(query)
        stages["classification_ms"] = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        expanded = self.expander.expand(query)
        stages["query_expansion_ms"] = (time.perf_counter() - stage_start) * 1000

        if classification.category == "FACTUAL_LOOKUP":
            pipeline = "Factual Lookup"
            fetch_k = settings.factual_fetch_k
        else:
            pipeline = "Broad / Strategic"
            fetch_k = settings.broad_fetch_k

        stage_start = time.perf_counter()
        candidates = self.retriever.retrieve(expanded, top_k=fetch_k)
        stages["retrieval_ms"] = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        ranked = self.reranker.rerank(query, candidates)
        stages["reranking_ms"] = (time.perf_counter() - stage_start) * 1000
        context_limit = settings.answer_context_chunks
        if classification.category == "FACTUAL_LOOKUP":
            context_limit = min(context_limit, settings.factual_retrieval_k)
        selected = ranked[:context_limit]

        stage_start = time.perf_counter()
        answer, answer_mode = self.generator.generate(query, selected)
        stages["generation_ms"] = (time.perf_counter() - stage_start) * 1000
        sources = list(dict.fromkeys(
            f"{item.source}, page {item.page}" if item.page is not None else item.source
            for item in selected
        ))
        result = QueryResult(
            query=query,
            answer=answer,
            classification=classification.category,
            classification_confidence=classification.confidence,
            pipeline=pipeline,
            cache_hit=False,
            latency_ms=(time.perf_counter() - started) * 1000,
            expanded_queries=expanded,
            retrieved_chunks=[asdict(item) for item in ranked],
            sources=sources,
            stage_timings_ms=stages,
            answer_mode=answer_mode,
        )
        stage_start = time.perf_counter()
        try:
            self.cache.set(query, result.to_dict())
        except Exception as exc:
            logger.warning("Cache write failed for request %s: %s", request_id, exc)
        stages["cache_write_ms"] = (time.perf_counter() - stage_start) * 1000
        result.stage_timings_ms = stages
        result.latency_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "request_id=%s classification=%s pipeline=%s cache=miss retrieved=%d reranked=%d latency_ms=%.1f",
            request_id, result.classification, pipeline, len(candidates), len(ranked), result.latency_ms,
        )
        return result
