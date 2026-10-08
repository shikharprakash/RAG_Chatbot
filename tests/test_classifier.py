from dataclasses import replace

from src.routing.classifier import QueryClassifier


def test_factual_error_lookup_uses_factual_fallback(monkeypatch):
    import src.routing.classifier as classifier_module
    from src.config import settings

    monkeypatch.setattr(classifier_module, "settings", replace(settings, grok_api_key=""))
    result = QueryClassifier().classify("What does error code F123 indicate?")
    assert result.category == "FACTUAL_LOOKUP"
    assert result.fallback_used


def test_broad_troubleshooting_uses_broad_fallback(monkeypatch):
    import src.routing.classifier as classifier_module
    from src.config import settings

    monkeypatch.setattr(classifier_module, "settings", replace(settings, grok_api_key=""))
    result = QueryClassifier().classify("What are possible causes and troubleshooting steps?")
    assert result.category == "BROAD_STRATEGIC"
    assert result.fallback_used
