import pytest

from lexisync.indexing import HybridIndex


class FakeBM25:
    def __init__(self, scores):
        self.scores = scores

    def get_scores(self, _tokens):
        return self.scores


class FakeCrossEncoder:
    def predict(self, pairs):
        return [1.0 if text == "distractor" else 0.0 for _, text in pairs]


class UnusedCrossEncoder:
    def predict(self, _pairs):
        raise AssertionError("reranker should not run when rerank=False")


def test_reranker_fusion_preserves_a_top_ranked_retrieval_candidate():
    index = HybridIndex.__new__(HybridIndex)
    index.chunks = [
        {"chunk_id": "gold", "text": "gold evidence"},
        {"chunk_id": "distractor", "text": "distractor"},
        {"chunk_id": "c", "text": "other passage"},
    ]
    index.bm25 = FakeBM25([3.0, 2.0, 1.0])
    index.embedder = None
    index.vectors = None
    index.cross = FakeCrossEncoder()

    results, _, trace = index.search("test question", top_k=1, mode="hybrid")

    assert results[0]["chunk_id"] == "gold"
    assert trace["reranker_fused"] is True
    assert trace["candidate_pool_size"] == 3


def test_dense_mode_does_not_silently_fall_back_to_bm25():
    index = HybridIndex.__new__(HybridIndex)
    index.chunks = [{"chunk_id": "a", "text": "evidence"}]
    index.bm25 = FakeBM25([1.0])
    index.embedder = None
    index.vectors = None
    index.cross = None

    with pytest.raises(RuntimeError, match="Dense retrieval is unavailable"):
        index.search("query", mode="dense")


def test_search_can_measure_retrieval_without_cross_encoder_reranking():
    index = HybridIndex.__new__(HybridIndex)
    index.chunks = [
        {"chunk_id": "first", "text": "first passage"},
        {"chunk_id": "second", "text": "second passage"},
    ]
    index.bm25 = FakeBM25([2.0, 1.0])
    index.embedder = None
    index.vectors = None
    index.cross = UnusedCrossEncoder()

    results, _, trace = index.search("query", top_k=1, mode="bm25", rerank=False)

    assert results[0]["chunk_id"] == "first"
    assert trace["reranker_enabled"] is True
    assert trace["reranker_fused"] is False
