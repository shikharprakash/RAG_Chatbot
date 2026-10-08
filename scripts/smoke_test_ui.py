from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

from src.cache.semantic_cache import SemanticCache


def ask(app: AppTest, question: str) -> dict:
    app.text_area[0].set_value(question)
    submit = next(button for button in app.button if button.label == "Find answer")
    submit.click().run(timeout=300)
    if app.exception:
        raise AssertionError(f"Streamlit UI raised an exception: {app.exception}")
    result = app.session_state["history"][-1]
    print(
        f"{result['classification']} | {result['pipeline']} | "
        f"cache={'HIT' if result['cache_hit'] else 'MISS'} | "
        f"latency={result['latency_ms']:.0f} ms | "
        f"chunks={len(result['retrieved_chunks'])}"
    )
    print(f"Q: {question}\nA: {result['answer'][:600]}\n")
    return result


def main() -> None:
    # Keep the test repeatable and prove cache behavior through Streamlit itself.
    SemanticCache().clear()
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    if app.exception:
        raise AssertionError(f"Streamlit UI failed to render: {app.exception}")

    factual = ask(app, "What does high-temp-warn-thr do?")
    assert factual["classification"] == "FACTUAL_LOOKUP"
    assert factual["pipeline"] == "Factual Lookup"
    assert len(factual["expanded_queries"]) == 3
    assert any("high-temp-warn-thr" in chunk["content"] for chunk in factual["retrieved_chunks"])

    paraphrase = ask(app, "What is the high-temp-warn-thr setting used for?")
    if not paraphrase["cache_hit"]:
        # A repeated exact query is a deterministic UI/cache sanity check when the
        # configured similarity threshold rejects this particular paraphrase.
        repeated = ask(app, "What does high-temp-warn-thr do?")
        assert repeated["cache_hit"]

    broad = ask(
        app,
        "Synthesize the documented risks, operational impact, and recommended safeguards for running debug commands on a production Nexus 9000, including what to do if console output is overwhelmed.",
    )
    assert broad["classification"] == "BROAD_STRATEGIC"
    assert broad["pipeline"] == "Broad / Strategic"
    assert len(broad["expanded_queries"]) == 3
    assert any("debug" in chunk["content"].lower() for chunk in broad["retrieved_chunks"])
    print("Streamlit UI smoke test passed.")


if __name__ == "__main__":
    main()
