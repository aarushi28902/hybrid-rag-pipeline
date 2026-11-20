from functools import lru_cache
from fastapi import Request
from qdrant_client import AsyncQdrantClient
from src.rag.config import get_settings


@lru_cache
def get_qdrant() -> AsyncQdrantClient:
    settings = get_settings()
    return AsyncQdrantClient(url=settings.qdrant_url)


def get_embedder(request: Request):
    return request.app.state.embedder


def get_settings_dep():
    return get_settings()


def get_reranker(request: Request):
    """The loaded cross-encoder, or None when reranking is disabled."""
    return getattr(request.app.state, "reranker", None)
