from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import truststore

truststore.inject_into_ssl()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import settings


_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def create_vectorstore(persist_directory: Path | None = None) -> Chroma:
    path = persist_directory or settings.chroma_persist_directory
    path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(path),
    )


def index_documents(documents: list[Document]) -> int:
    if not documents:
        raise ValueError("No document chunks were provided for indexing.")
    store = create_vectorstore()
    ids = [str(document.metadata["chunk_id"]) for document in documents]
    store.add_documents(documents, ids=ids)
    return len(documents)


def collection_count() -> int:
    try:
        return int(create_vectorstore()._collection.count())
    except Exception:
        return 0
