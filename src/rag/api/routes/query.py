import json
import time

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.rag.api.dependencies import get_embedder, get_qdrant, get_reranker
from src.rag.config import get_settings
from src.rag.generation.generator import StreamingGenerator
from src.rag.monitoring.metrics import (
    ABSTENTIONS,
    ACTIVE_REQUESTS,
    GENERATION_LATENCY,
    QUERY_ERRORS,
    REQUEST_LATENCY,
    RERANK_LATENCY,
    RETRIEVAL_LATENCY,
)
from src.rag.retrieval.hybrid_search import HybridSearcher, SearchResult
from src.rag.retrieval.reranker import should_abstain

logger = structlog.get_logger()
router = APIRouter()

ABSTAIN_MESSAGE = "I couldn't find this in the documents."


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=20)
    collection: str = "enterprise_rag"


def sources_event(chunks: list[SearchResult]) -> str:
    """One SSE event listing what each [n] citation in the answer refers to."""
    sources = [
        {
            "n": i,
            "source": c.source,
            "parent_id": c.parent_id,
            "score": round(c.rerank_score, 3) if c.rerank_score is not None else None,
            "preview": c.text[:200],
        }
        for i, c in enumerate(chunks, start=1)
    ]
    return f"data: [SOURCES] {json.dumps(sources, ensure_ascii=False)}\n\n"


@router.post("/stream")
async def query_stream(
    req: QueryRequest,
    embedder=Depends(get_embedder),
    qdrant=Depends(get_qdrant),
    reranker=Depends(get_reranker),
):
    settings = get_settings()

    async def stream():
        ACTIVE_REQUESTS.inc()
        t0 = time.perf_counter()
        try:
            # retrieval: more candidates when a reranker will narrow them down
            t_ret = time.perf_counter()
            searcher = HybridSearcher(qdrant_client=qdrant, embedder=embedder)
            n_candidates = max(settings.rerank_candidates, req.top_k) if reranker else req.top_k
            chunks = await searcher.search(
                query=req.query,
                collection=req.collection,
                top_k=n_candidates,
            )
            RETRIEVAL_LATENCY.observe(time.perf_counter() - t_ret)

            if reranker and chunks:
                t_rr = time.perf_counter()
                chunks = await reranker.rerank(req.query, chunks, top_k=req.top_k)
                RERANK_LATENCY.observe(time.perf_counter() - t_rr)

            logger.info(
                "retrieval_done",
                n_chunks=len(chunks),
                best_score=chunks[0].rerank_score if chunks else None,
                query=req.query[:80],
            )

            # abstain instead of letting the model guess from weak context
            if should_abstain(chunks, settings.abstain_threshold):
                ABSTENTIONS.inc()
                logger.info("abstained", query=req.query[:80])
                yield f"data: {ABSTAIN_MESSAGE}\n\n"
                yield "data: [SOURCES] []\n\n"
                yield "data: [DONE]\n\n"
                REQUEST_LATENCY.observe(time.perf_counter() - t0)
                return

            # generation
            t_gen = time.perf_counter()
            generator = StreamingGenerator(
                model=settings.generation_model,
                base_url=f"{settings.ollama_url}/v1",
            )
            async for token in generator.stream(query=req.query, chunks=chunks):
                yield f"data: {token}\n\n"
            GENERATION_LATENCY.observe(time.perf_counter() - t_gen)

            yield sources_event(chunks)
            yield "data: [DONE]\n\n"
            REQUEST_LATENCY.observe(time.perf_counter() - t0)

        except Exception as e:
            QUERY_ERRORS.inc()
            logger.error("query_failed", error=str(e))
            yield "data: [ERROR] Something went wrong while answering. Please try again.\n\n"

        finally:
            ACTIVE_REQUESTS.dec()

    return StreamingResponse(stream(), media_type="text/event-stream")
