from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger(__name__)


class _ExpansionSchema(BaseModel):
    expanded_queries: list[str] = Field(description="Exactly three useful, intent-preserving retrieval variations.")


class QueryExpander:
    def __init__(self, llm=None, count: int | None = None) -> None:
        self.llm = llm
        self.count = count or settings.query_expansion_count

    def expand(self, query: str) -> list[str]:
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
            structured = llm.with_structured_output(_ExpansionSchema)
            result = structured.invoke([
                ("system", f"Produce exactly {self.count} concise search-query variations. Preserve all device names, identifiers, error codes, and original intent. Do not answer the question or invent terminology."),
                ("human", query),
            ])
            expansions = [text.strip() for text in result.expanded_queries if text.strip()]
            if len(expansions) != self.count:
                raise ValueError(f"Expected {self.count} query variations, received {len(expansions)}")
            return expansions
        except Exception as exc:
            logger.warning("Query expansion failed; using deterministic fallback: %s", exc)
            return self._fallback(query)

    def _fallback(self, query: str) -> list[str]:
        original = query.strip().rstrip("? ")
        variations = [
            f"Cisco Nexus 9000 {original}",
            f"{original} documented meaning and relevant details",
            f"Cisco Nexus 9000 troubleshooting guide: {original} causes, requirements, or resolution",
        ]
        return variations[: self.count]
