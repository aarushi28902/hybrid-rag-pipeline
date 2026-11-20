from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.rag.ingestion.chunker import ParentChildChunker
from src.rag.ingestion.pipeline import build_points
from src.rag.retrieval.embedder import SparseEmbedding
from src.rag.retrieval.hybrid_search import (
    RRF_K,
    HybridSearcher,
    collapse_to_parents,
    rrf_fuse,
)


# ---------- chunker IDs ----------

def test_parent_ids_are_deterministic():
    chunker = ParentChildChunker(parent_size=200, child_size=50, overlap=10)
    text = "The quick brown fox jumps over the lazy dog. " * 20
    ids1 = [p.chunk_id for p, _ in chunker.chunk(text, {"source": "a"})]
    ids2 = [p.chunk_id for p, _ in chunker.chunk(text, {"source": "a"})]
    assert ids1 == ids2


def test_same_text_in_different_sources_gets_different_ids():
    chunker = ParentChildChunker(parent_size=200, child_size=50, overlap=10)
    text = "word " * 100
    a = {c.chunk_id for _, cs in chunker.chunk(text, {"source": "a"}) for c in cs}
    b = {c.chunk_id for _, cs in chunker.chunk(text, {"source": "b"}) for c in cs}
    assert a.isdisjoint(b)


def test_repeated_child_text_does_not_collide():
    # "word " repeated gives identical child texts in every parent;
    # each child must still get its own ID.
    chunker = ParentChildChunker(parent_size=200, child_size=50, overlap=10)
    children = [c for _, cs in chunker.chunk("word " * 300, {"source": "a"}) for c in cs]
    assert len({c.chunk_id for c in children}) == len(children)


# ---------- RRF ----------

def test_rrf_scores_match_formula():
    fused = dict(rrf_fuse([["a", "b"], ["b", "c"]]))
    assert fused["a"] == pytest.approx(1 / (RRF_K + 1))
    assert fused["b"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert fused["c"] == pytest.approx(1 / (RRF_K + 2))


def test_rrf_rewards_agreement_over_a_single_first_place():
    # "b" is second in both lists; "a" is first in only one.
    order = [pid for pid, _ in rrf_fuse([["a", "b"], ["c", "b"]])]
    assert order[0] == "b"


def test_rrf_single_list_keeps_order():
    assert [p for p, _ in rrf_fuse([["x", "y", "z"]])] == ["x", "y", "z"]


# ---------- parent collapsing ----------

def payload(child, parent, parent_text="P"):
    return {"text": f"child {child}", "chunk_id": child,
            "parent_id": parent, "parent_text": parent_text}


def test_collapse_returns_each_parent_once_in_rank_order():
    fused = [("c1", 0.9), ("c2", 0.8), ("c3", 0.7)]
    payloads = {
        "c1": payload("c1", "p1", "parent one"),
        "c2": payload("c2", "p1", "parent one"),
        "c3": payload("c3", "p2", "parent two"),
    }
    results = collapse_to_parents(fused, payloads, top_k=5)
    assert [r.parent_id for r in results] == ["p1", "p2"]
    assert results[0].text == "parent one"
    assert results[0].matched_text == "child c1"


def test_collapse_respects_top_k():
    fused = [(f"c{i}", 1 - i / 10) for i in range(5)]
    payloads = {f"c{i}": payload(f"c{i}", f"p{i}") for i in range(5)}
    assert len(collapse_to_parents(fused, payloads, top_k=2)) == 2


def test_collapse_falls_back_to_child_text_for_old_points():
    payloads = {"c1": {"text": "only child", "chunk_id": "c1", "parent_id": "p1"}}
    results = collapse_to_parents([("c1", 1.0)], payloads, top_k=3)
    assert results[0].text == "only child"


# ---------- searcher modes ----------

def point(pid, parent):
    return SimpleNamespace(id=pid, payload=payload(pid, parent, f"text of {parent}"))


@pytest.fixture
def searcher():
    embedder = AsyncMock()
    embedder.embed_query.return_value = ([0.1] * 4, SparseEmbedding([1], [0.5]))
    qdrant = AsyncMock()

    async def fake_search(collection_name, query_vector, limit, with_payload):
        if query_vector.name == "dense":
            return [point("d1", "p1"), point("s1", "p2")]
        return [point("s1", "p2"), point("s2", "p3")]

    qdrant.search.side_effect = fake_search
    return HybridSearcher(qdrant_client=qdrant, embedder=embedder)


@pytest.mark.asyncio
async def test_hybrid_queries_both_indexes(searcher):
    results = await searcher.search("q", "col", top_k=6)
    assert searcher.qdrant.search.await_count == 2
    # s1 is found by both retrievers, so its parent ranks first.
    assert results[0].parent_id == "p2"
    assert {r.parent_id for r in results} == {"p1", "p2", "p3"}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,expected", [("dense", ["p1", "p2"]), ("sparse", ["p2", "p3"])])
async def test_single_mode_queries_one_index(searcher, mode, expected):
    results = await searcher.search("q", "col", top_k=6, mode=mode)
    assert searcher.qdrant.search.await_count == 1
    assert [r.parent_id for r in results] == expected


# ---------- ingestion payload ----------

@pytest.mark.asyncio
async def test_build_points_attach_parent_text():
    chunker = ParentChildChunker(parent_size=200, child_size=50, overlap=10)
    pairs = list(chunker.chunk("The quick brown fox. " * 30, {"source": "doc"}))

    embedder = AsyncMock()
    embedder.embed_documents.side_effect = lambda texts: [
        ([0.1] * 4, SparseEmbedding([1], [0.5])) for _ in texts
    ]

    points = await build_points(pairs, embedder, source="doc")
    assert len(points) == sum(len(cs) for _, cs in pairs)
    parents = {parent.chunk_id: parent.text for parent, _ in pairs}
    for p in points:
        assert p.payload["parent_text"] == parents[p.payload["parent_id"]]
        assert p.payload["source"] == "doc"
    # One embedding call per parent, not one per child.
    assert embedder.embed_documents.await_count == len(pairs)
