import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_embedder():
    emb = AsyncMock()
    emb.embed_query.return_value = (
        [0.1] * 1024,
        MagicMock(indices=[1, 2], values=[0.5, 0.3]),
    )
    return emb


@pytest_asyncio.fixture
async def client(mock_embedder):
    from src.rag.api.main import create_app
    from src.rag.api.dependencies import get_embedder, get_qdrant

    app = create_app()
    app.state.embedder = mock_embedder

    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_qdrant] = lambda: AsyncMock()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_health_check(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_query_too_short_rejected(client):
    r = await client.post(
        "/api/v1/query/stream",
        json={"query": "hi"},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_query_valid_returns_stream(client):
    r = await client.post(
        "/api/v1/query/stream",
        json={"query": "What is retrieval augmented generation?"},
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]


# ---------- reranking, abstention and citations ----------

from src.rag.retrieval.hybrid_search import SearchResult


def _result(pid, source="doc.txt"):
    return SearchResult(chunk_id=pid, parent_id=pid, text=f"passage {pid}", score=0.1, source=source)


class FakeSearcher:
    def __init__(self, qdrant_client=None, embedder=None):
        pass

    async def search(self, query, collection, top_k, mode="hybrid"):
        return [_result(f"p{i}") for i in range(top_k)]


class FakeReranker:
    def __init__(self, best):
        self.best = best

    async def rerank(self, query, results, top_k):
        from src.rag.retrieval.reranker import rerank_by_scores
        scores = [self.best - i * 0.01 for i in range(len(results))]
        return rerank_by_scores(results, scores, top_k)


class FakeGenerator:
    calls = 0

    def __init__(self, **kwargs):
        pass

    async def stream(self, query, chunks):
        FakeGenerator.calls += 1
        for token in ["Answer ", "[1]"]:
            yield token


async def _stream_with(monkeypatch, mock_embedder, best_score):
    from src.rag.api.main import create_app
    from src.rag.api.dependencies import get_embedder, get_qdrant, get_reranker
    import src.rag.api.routes.query as query_route

    monkeypatch.setattr(query_route, "HybridSearcher", FakeSearcher)
    monkeypatch.setattr(query_route, "StreamingGenerator", FakeGenerator)
    FakeGenerator.calls = 0

    app = create_app()
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_qdrant] = lambda: AsyncMock()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker(best_score)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/query/stream", json={"query": "Who produced Folklore?"})
    return r.text


@pytest.mark.asyncio
async def test_low_confidence_abstains_without_calling_the_llm(monkeypatch, mock_embedder):
    body = await _stream_with(monkeypatch, mock_embedder, best_score=0.001)
    assert "couldn't find this in the documents" in body
    assert "data: [SOURCES] []" in body
    assert FakeGenerator.calls == 0


@pytest.mark.asyncio
async def test_confident_answer_streams_with_citations(monkeypatch, mock_embedder):
    import json

    body = await _stream_with(monkeypatch, mock_embedder, best_score=0.9)
    assert FakeGenerator.calls == 1
    assert "data: Answer " in body
    line = next(l for l in body.splitlines() if l.startswith("data: [SOURCES]"))
    sources = json.loads(line.removeprefix("data: [SOURCES] "))
    assert len(sources) == 6  # default top_k after reranking 30 candidates
    assert sources[0]["n"] == 1 and sources[0]["score"] == 0.9
    assert body.rstrip().endswith("data: [DONE]")


@pytest.mark.asyncio
async def test_errors_do_not_leak_exception_text(monkeypatch, mock_embedder):
    from src.rag.api.main import create_app
    from src.rag.api.dependencies import get_embedder, get_qdrant, get_reranker
    import src.rag.api.routes.query as query_route

    class Broken(FakeSearcher):
        async def search(self, *a, **k):
            raise RuntimeError("secret connection string")

    monkeypatch.setattr(query_route, "HybridSearcher", Broken)
    app = create_app()
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_qdrant] = lambda: AsyncMock()
    app.dependency_overrides[get_reranker] = lambda: None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/query/stream", json={"query": "Who produced Folklore?"})
    assert "[ERROR]" in r.text and "secret" not in r.text
