from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from src.config import settings
from src.models import ClassificationResult

logger = logging.getLogger(__name__)


class _ClassificationSchema(BaseModel):
    category: str = Field(description="FACTUAL_LOOKUP or BROAD_STRATEGIC")
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class QueryClassifier:
    def __init__(self, llm=None) -> None:
        self.llm = llm

    def classify(self, query: str) -> ClassificationResult:
        if not query.strip():
            return ClassificationResult("BROAD_STRATEGIC", 0.0, "Empty query.", True)
        try:
            if not settings.grok_api_key:
                raise RuntimeError("GROK_API_KEY is not configured")
            llm = self.llm
            if llm is None:
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(
                    model=settings.llm_model,
                    temperature=0,
                    api_key=settings.grok_api_key,
                    base_url=settings.llm_base_url,
                )
            structured = llm.with_structured_output(_ClassificationSchema)
            result = structured.invoke([
                ("system", "Classify as FACTUAL_LOOKUP for a narrow fact, definition, code, specification, or direct requirement. Use BROAD_STRATEGIC for multiple causes, steps, comparisons, or synthesis. Return only the schema."),
                ("human", query),
            ])
            category = str(result.category).upper()
            if category not in {"FACTUAL_LOOKUP", "BROAD_STRATEGIC"}:
                raise ValueError(f"Invalid classification category: {category}")
            return ClassificationResult(category, float(result.confidence), result.reason)
        except Exception as exc:
            logger.warning("Classification failed; applying conservative fallback: %s", exc)
            return self._fallback(query)

    @staticmethod
    def _fallback(query: str) -> ClassificationResult:
        broad_pattern = r"\b(steps?|causes?|troubleshoot|possible|multiple|compare|and|why does .* keep|repeatedly|recommendations?)\b"
        factual_pattern = r"\b(what is|what does|meaning of|define|error code|status code|how many|which command|specification|value of)\b"
        is_broad = bool(re.search(broad_pattern, query, re.IGNORECASE))
        is_factual = bool(re.search(factual_pattern, query, re.IGNORECASE))
        category = "FACTUAL_LOOKUP" if is_factual and not is_broad else "BROAD_STRATEGIC"
        confidence = 0.7 if category == "FACTUAL_LOOKUP" else 0.6
        return ClassificationResult(category, confidence, "Rule-based fallback used because model classification was unavailable.", True)
