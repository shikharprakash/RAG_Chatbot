from dataclasses import replace

from src.retrieval.query_expander import QueryExpander


def test_fallback_expansion_returns_exactly_three_intent_related_queries(monkeypatch):
    import src.retrieval.query_expander as expansion_module
    from src.config import settings

    monkeypatch.setattr(expansion_module, "settings", replace(settings, grok_api_key=""))
    expanded = QueryExpander().expand("What does error F123 mean?")
    assert len(expanded) == 3
    assert all("F123" in query for query in expanded)
