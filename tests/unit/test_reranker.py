import pytest

from src.rag.retrieval.hybrid_search import SearchResult
from src.rag.retrieval.reranker import Reranker, rerank_by_scores, should_abstain


def result(pid: str, score: float | None = None) -> SearchResult:
    return SearchResult(chunk_id=pid, parent_id=pid, text=f"text {pid}", score=0.0, rerank_score=score)


def test_rerank_by_scores_orders_by_score_and_keeps_top_k():
    out = rerank_by_scores([result("a"), result("b"), result("c")], [0.2, 0.9, 0.5], top_k=2)
    assert [r.parent_id for r in out] == ["b", "c"]
    assert [r.rerank_score for r in out] == [0.9, 0.5]


def test_rerank_does_not_mutate_inputs():
    original = [result("a")]
    rerank_by_scores(original, [0.7], top_k=1)
    assert original[0].rerank_score is None


def test_should_abstain_below_threshold():
    assert should_abstain([result("a", 0.02)], threshold=0.05)
    assert not should_abstain([result("a", 0.30)], threshold=0.05)


def test_should_abstain_when_nothing_retrieved():
    assert should_abstain([], threshold=0.05)


def test_no_abstention_without_reranker_scores():
    # Without a reranker there is no calibrated score, so never abstain on score.
    assert not should_abstain([result("a", None)], threshold=0.9)


@pytest.mark.asyncio
async def test_rerank_uses_model_scores(monkeypatch):
    reranker = Reranker()
    monkeypatch.setattr(reranker, "score_sync", lambda q, passages: [0.1, 0.8, 0.4])
    out = await reranker.rerank("q", [result("a"), result("b"), result("c")], top_k=3)
    assert [r.parent_id for r in out] == ["b", "c", "a"]
