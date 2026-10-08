from __future__ import annotations

import math

from langchain_chroma import Chroma

from src.models import RetrievedChunk
from src.vectorstore import create_vectorstore


class MultiQueryRetriever:
    def __init__(self, vectorstore: Chroma | None = None) -> None:
        self.vectorstore = vectorstore or create_vectorstore()

    def retrieve(self, queries: list[str], top_k: int) -> list[RetrievedChunk]:
        if not queries or top_k <= 0:
            return []
        per_query = max(4, math.ceil(top_k / len(queries)) * 2)
        unique: dict[str, RetrievedChunk] = {}
        for query in queries:
            for document, distance in self.vectorstore.similarity_search_with_score(
                query, k=per_query
            ):
                metadata = dict(document.metadata)
                chunk_id = str(metadata.get("chunk_id") or metadata.get("id") or "")
                if not chunk_id:
                    import hashlib

                    chunk_id = hashlib.sha1(document.page_content.encode()).hexdigest()
                page = metadata.get("page")
                item = RetrievedChunk(
                    chunk_id=chunk_id,
                    content=document.page_content,
                    source=str(metadata.get("source", metadata.get("document_name", "unknown"))),
                    page=int(page) if page is not None else None,
                    retrieval_score=1.0 / (1.0 + max(float(distance), 0.0)),
                    metadata=metadata,
                )
                previous = unique.get(chunk_id)
                if previous is None or item.retrieval_score > previous.retrieval_score:
                    unique[chunk_id] = item
        return sorted(unique.values(), key=lambda item: item.retrieval_score, reverse=True)[:top_k]
