from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _path_value(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default)).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


def _provider_api_key() -> str:
    return os.getenv("GROK_API_KEY", "").strip() or os.getenv("GROQ_API_KEY", "").strip()


@dataclass(frozen=True)
class Settings:
    grok_api_key: str = _provider_api_key()
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    source_document_path: Path = _path_value(
        "SOURCE_DOCUMENT_PATH",
        "cisco-nexus-9000-series-nx-os-troubleshooting-guide-106x.pdf",
    )
    chroma_persist_directory: Path = _path_value(
        "CHROMA_PERSIST_DIRECTORY", "data/chroma"
    )
    collection_name: str = os.getenv(
        "COLLECTION_NAME", "cisco_nexus_troubleshooting"
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1024"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "32"))
    factual_retrieval_k: int = int(os.getenv("FACTUAL_RETRIEVAL_K", "4"))
    broad_retrieval_k: int = int(os.getenv("BROAD_RETRIEVAL_K", "20"))
    factual_fetch_k: int = int(os.getenv("FACTUAL_FETCH_K", "12"))
    broad_fetch_k: int = int(os.getenv("BROAD_FETCH_K", "20"))
    query_expansion_count: int = int(os.getenv("QUERY_EXPANSION_COUNT", "3"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    cache_similarity_threshold: float = float(
        os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.88")
    )
    cache_file_path: Path = _path_value(
        "CACHE_FILE_PATH", "data/cache/semantic_cache.json"
    )
    answer_context_chunks: int = int(os.getenv("ANSWER_CONTEXT_CHUNKS", "6"))


settings = Settings()
