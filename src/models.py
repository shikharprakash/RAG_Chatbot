from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Category = Literal["FACTUAL_LOOKUP", "BROAD_STRATEGIC"]


@dataclass
class ClassificationResult:
    category: Category
    confidence: float
    reason: str
    fallback_used: bool = False


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    source: str
    page: int | None
    retrieval_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankedChunk(RetrievedChunk):
    rerank_score: float = 0.0
    final_rank: int = 0


@dataclass
class QueryResult:
    query: str
    answer: str
    classification: str
    classification_confidence: float
    pipeline: str
    cache_hit: bool
    latency_ms: float
    expanded_queries: list[str] = field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    answer_mode: str = "openai"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QueryResult":
        return cls(**value)
