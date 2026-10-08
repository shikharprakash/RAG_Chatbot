from src.models import ClassificationResult, RetrievedChunk, RankedChunk
from src.orchestration.query_orchestrator import QueryOrchestrator


class FakeCache:
    def __init__(self):
        self.value = None
        self.hits = 0
        self.misses = 0

    def get(self, query):
        if self.value:
            self.hits += 1
            return self.value
        self.misses += 1
        return None

    def set(self, query, result):
        self.value = result


class FakeClassifier:
    def classify(self, query):
        category = "FACTUAL_LOOKUP" if "error code" in query else "BROAD_STRATEGIC"
        return ClassificationResult(category, 0.99, "test")


class FakeExpander:
    def expand(self, query):
        return [f"{query} variation {number}" for number in range(1, 4)]


class FakeRetriever:
    def __init__(self):
        self.last_k = None

    def retrieve(self, queries, top_k):
        self.last_k = top_k
        return [RetrievedChunk("1", "Document evidence", "guide.pdf", 7, 0.8)]


class FakeReranker:
    def rerank(self, query, chunks):
        return [RankedChunk(**chunks[0].__dict__, rerank_score=0.9, final_rank=1)]


class FakeGenerator:
    def generate(self, query, context):
        return "Grounded test answer", "test"


def test_factual_routing_and_cache_hit():
    cache = FakeCache()
    retriever = FakeRetriever()
    orchestrator = QueryOrchestrator(cache, FakeClassifier(), FakeExpander(), retriever, FakeReranker(), FakeGenerator())
    first = orchestrator.answer("What does error code F123 indicate?")
    assert first.classification == "FACTUAL_LOOKUP"
    assert first.pipeline == "Factual Lookup"
    assert first.cache_hit is False
    assert len(first.expanded_queries) == 3
    assert first.sources == ["guide.pdf, page 7"]
    assert retriever.last_k > 0

    second = orchestrator.answer("What does error code F123 indicate?")
    assert second.cache_hit is True
    assert second.answer == first.answer


def test_broad_query_routes_to_broad_pipeline():
    retriever = FakeRetriever()
    orchestrator = QueryOrchestrator(FakeCache(), FakeClassifier(), FakeExpander(), retriever, FakeReranker(), FakeGenerator())
    result = orchestrator.answer("What are possible causes and troubleshooting steps?")
    assert result.classification == "BROAD_STRATEGIC"
    assert result.pipeline == "Broad / Strategic"
