from contextlib import asynccontextmanager
import structlog
from fastapi import FastAPI
from prometheus_client import make_asgi_app

from src.rag.api.routes import ingest, query
from src.rag.config import get_settings
from src.rag.retrieval.embedder import EmbedderService
from src.rag.retrieval.reranker import Reranker

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup: loading embedder")
    app.state.embedder = EmbedderService()
    await app.state.embedder.load()
    settings = get_settings()
    app.state.reranker = None
    if settings.rerank_enabled:
        logger.info("startup: loading reranker", model=settings.rerank_model)
        app.state.reranker = Reranker(settings.rerank_model)
        await app.state.reranker.load()
    logger.info("startup: complete")
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise RAG API",
        description="Hybrid search RAG with streaming and observability",
        version="0.1.0",
        lifespan=lifespan,
    )

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingestion"])
    app.include_router(query.router, prefix="/api/v1/query", tags=["query"])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
