from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.pdf_loader import load_pdf, split_pages
from src.vectorstore import index_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    pages = load_pdf()
    chunks = split_pages(pages)
    count = index_documents(chunks)
    logging.info("Indexed %s chunks from %s extracted pages.", count, len(pages))


if __name__ == "__main__":
    main()
