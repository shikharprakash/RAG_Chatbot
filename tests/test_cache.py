import time

from src.cache.semantic_cache import SemanticCache


def test_semantic_cache_hit_and_expiration(tmp_path):
    vectors = {"original question": [1.0, 0.0], "similar question": [0.99, 0.01]}
    cache = SemanticCache(
        ttl_seconds=10,
        similarity_threshold=0.95,
        file_path=tmp_path / "cache.json",
        embed_query=lambda query: vectors[query],
    )
    cache.set("original question", {"answer": "cached"})
    assert cache.get("similar question")["answer"] == "cached"
    assert cache.metrics()["cache_hits"] == 1

    cache.ttl_seconds = 0
    cache.clear()
    cache.set("original question", {"answer": "expired"})
    time.sleep(0.01)
    assert cache.get("original question") is None
