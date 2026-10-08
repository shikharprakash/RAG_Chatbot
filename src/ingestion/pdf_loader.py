from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import settings


def load_pdf(path: Path | None = None) -> list[Document]:
    pdf_path = path or settings.source_document_path
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Source PDF was not found: {pdf_path}")
    try:
        pages = PyPDFLoader(str(pdf_path)).load()
    except Exception as exc:
        raise RuntimeError(f"Could not extract pages from PDF {pdf_path}: {exc}") from exc
    if not pages or not any(page.page_content.strip() for page in pages):
        raise ValueError(f"The PDF contains no extractable text: {pdf_path}")
    for page in pages:
        page.metadata["source"] = pdf_path.name
        page.metadata["document_name"] = pdf_path.name
        if "page" in page.metadata:
            page.metadata["page"] = int(page.metadata["page"]) + 1
    return pages


def split_pages(
    pages: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=settings.chunk_overlap if chunk_overlap is None else chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'document')}:{index}"
    return chunks
