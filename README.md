# Cisco Nexus 9000 Troubleshooting Assistant

A local Streamlit RAG proof of concept following `PRD.md`. It supports PDF ingestion, persistent Chroma retrieval, three-query expansion, routed factual/broad retrieval, cross-encoder reranking, a semantic cache with configurable TTL, answer generation, and visible pipeline diagnostics.

## 1. Environment and installation

Python 3.12 is recommended. In this workspace the virtual environment is `.venv`.

```zsh
source .venv/bin/activate
cp .env.example .env
# Add your GROK_API_KEY to .env to enable openai/gpt-oss-120b via Groq.
pip install -r requirements.txt
```

Without an API key, rule-based classification and deterministic query expansion are used. If the cross-encoder cannot be downloaded/loaded, reranking falls back to lexical relevance; without Groq, retrieved passages are shown extractively rather than passed off as an LLM answer. The app uses `GROK_API_KEY` as requested and also accepts `GROQ_API_KEY` for compatibility with an existing local environment. Embedding and reranker model files are downloaded on first use and run on CPU.

## 2. Index the troubleshooting PDF

The supplied PDF in the project root is the default source. Index it once (this downloads the local embedding model the first time):

```zsh
python scripts/ingest_document.py
```

The persistent Chroma database is stored under `data/chroma/`. Re-run ingestion when replacing the source PDF. `SOURCE_DOCUMENT_PATH` and `CHROMA_PERSIST_DIRECTORY` can be overridden in `.env`.

## 3. Run the UI

```zsh
streamlit run app.py
```

The sidebar reports index status and provides current-session conversation history. Each response displays classification, selected pipeline, cache hit/miss, latency, ranked source passages, expansion queries, stage timings, and cache metrics.

## Configuration

See `.env.example`. Key settings include `LLM_MODEL`, `EMBEDDING_MODEL`, `RERANKER_MODEL`, chunk size/overlap, factual/broad retrieval counts, semantic-cache threshold, and TTL. Secrets should only be set in `.env` or Streamlit secrets, never committed.

## Tests

```zsh
pytest -q
python scripts/smoke_test_ui.py
```

The pytest suite uses deterministic fakes. The Streamlit UI smoke test submits factual and broad questions against the local PDF index and checks routing, source evidence, and semantic-cache reuse; it requires the indexed collection and local embedding/reranker model files. Without a Groq key, the app uses safe classification/expansion fallbacks and returns retrieved source excerpts without claiming an LLM-generated interpretation. Add `GROK_API_KEY` to `.env` to activate `openai/gpt-oss-120b` classification, query expansion, and answer generation through the Groq-compatible API.
