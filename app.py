from __future__ import annotations

import logging

import streamlit as st

from src.config import settings
from src.orchestration.query_orchestrator import QueryOrchestrator
from src.vectorstore import collection_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

st.set_page_config(page_title="Cisco Nexus 9000 Troubleshooting Assistant", page_icon="🛠️", layout="wide")


@st.cache_resource
def get_orchestrator() -> QueryOrchestrator:
    return QueryOrchestrator()


@st.cache_data(ttl=15)
def get_index_count() -> int:
    return collection_count()


if "history" not in st.session_state:
    st.session_state.history = []
if "active_result" not in st.session_state:
    st.session_state.active_result = None

st.title("Cisco Nexus 9000 Troubleshooting Assistant")
st.caption("Document-grounded troubleshooting with routed retrieval, reranking, and semantic caching")

with st.sidebar:
    st.header("Conversation history")
    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.active_result = None
        st.rerun()
    for index, item in enumerate(reversed(st.session_state.history)):
        original_index = len(st.session_state.history) - index - 1
        title = item["query"][:48] + ("…" if len(item["query"]) > 48 else "")
        if st.button(title, key=f"history_{original_index}", use_container_width=True):
            st.session_state.active_result = original_index
    st.divider()
    st.subheader("Knowledge base")
    try:
        chunk_count = get_index_count()
        if chunk_count:
            st.success(f"Indexed chunks: {chunk_count:,}")
        else:
            st.warning("No vector index found. Run `python scripts/ingest_document.py` first.")
    except Exception as exc:
        st.warning(f"Vector index unavailable: {exc}")
    if not settings.grok_api_key:
        st.info("GROK_API_KEY is not configured. Classification and query expansion use safe local fallbacks, and answers show retrieved source excerpts. Add GROK_API_KEY to .env to enable openai/gpt-oss-120b through the Groq API.")

left, right = st.columns([1.15, 1])
with left:
    with st.form("question_form", clear_on_submit=False):
        query = st.text_area(
            "Ask a Cisco Nexus 9000 troubleshooting question",
            placeholder="For example: What does error code XYZ indicate?",
            height=100,
        )
        submitted = st.form_submit_button("Find answer", type="primary", use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning("Enter a question to continue.")
        else:
            try:
                with st.spinner("Checking cache, routing, retrieving, and ranking evidence…"):
                    result = get_orchestrator().answer(query)
                st.session_state.history.append(result.to_dict())
                st.session_state.active_result = len(st.session_state.history) - 1
            except Exception as exc:
                st.error(f"The request could not be completed: {exc}")

active_index = st.session_state.active_result
result = (
    st.session_state.history[active_index]
    if isinstance(active_index, int) and 0 <= active_index < len(st.session_state.history)
    else None
)

with right:
    st.subheader("Answer")
    if result:
        st.markdown(result["answer"])
        if result.get("answer_mode") == "local_extractive":
            st.caption("Local extractive fallback — source passages are shown without LLM interpretation.")
        elif result.get("answer_mode") == "no_context":
            st.warning("No supporting context was retrieved; no answer was generated.")
    else:
        st.info("Submit a question to view a grounded answer and its retrieval diagnostics.")

if result:
    st.divider()
    metrics = st.columns(4)
    metrics[0].metric("Classification", result["classification"])
    metrics[1].metric("Selected pipeline", result["pipeline"])
    metrics[2].metric("Semantic cache", "HIT" if result["cache_hit"] else "MISS")
    metrics[3].metric("Total latency", f"{result['latency_ms']:.0f} ms")
    st.caption(f"Classification confidence: {result['classification_confidence']:.2f} · Answer mode: {result.get('answer_mode', 'groq')}")

    if result.get("sources"):
        st.markdown("**Sources:** " + " · ".join(result["sources"]))

    with st.expander("Retrieved and reranked context", expanded=True):
        chunks = result.get("retrieved_chunks", [])
        if not chunks:
            st.info("No relevant chunks were retrieved.")
        for chunk in chunks:
            page_text = f" · Page {chunk['page']}" if chunk.get("page") is not None else " · Page unavailable"
            title = f"#{chunk.get('final_rank', '?')} · rerank {chunk.get('rerank_score', 0):.3f} · retrieval {chunk.get('retrieval_score', 0):.3f}{page_text}"
            with st.expander(title):
                st.caption(chunk.get("source", "Unknown source"))
                st.text(chunk.get("content", ""))

    with st.expander("Pipeline diagnostics"):
        st.markdown("**Expanded retrieval queries**")
        for expanded_query in result.get("expanded_queries", []):
            st.markdown(f"- {expanded_query}")
        st.markdown("**Stage timings (ms)**")
        st.json(result.get("stage_timings_ms", {}))
        try:
            st.markdown("**Semantic cache metrics**")
            st.json(get_orchestrator().cache.metrics())
        except Exception as exc:
            st.caption(f"Cache metrics unavailable: {exc}")

with st.sidebar:
    if st.session_state.history:
        st.divider()
        st.caption(f"This session: {len(st.session_state.history)} question(s)")
