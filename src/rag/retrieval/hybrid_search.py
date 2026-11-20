from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import NamedSparseVector, NamedVector, SparseVector

from src.rag.retrieval.embedder import EmbedderService

RRF_K = 60
SearchMode = Literal["hybrid", "dense", "sparse"]


@dataclass
class SearchResult:
    chunk_id: str  # the best-matching child chunk
    parent_id: str
    text: str  # parent text: what the LLM sees
    score: float
    matched_text: str = ""  # the child text that actually matched
    source: str = ""
    rerank_score: float | None = None  # cross-encoder relevance in [0, 1], if reranked


def rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: each ID scores sum(1 / (k + rank)) across lists."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, pid in enumerate(ranked, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def collapse_to_parents(
    fused: list[tuple[str, float]], payloads: dict[str, dict], top_k: int
) -> list[SearchResult]:
    """Walk the fused child ranking and keep the first hit for each parent.

    Several children of one parent often match the same query; returning the
    parent once avoids filling the context window with the same passage.
    """
    results: list[SearchResult] = []
    seen_parents: set[str] = set()
    for pid, score in fused:
        payload = payloads.get(pid, {})
        child_text = payload.get("text", "")
        parent_id = payload.get("parent_id") or pid
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)
        results.append(
            SearchResult(
                chunk_id=payload.get("chunk_id", pid),
                parent_id=parent_id,
                # Points ingested before parent_text existed fall back to the child.
                text=payload.get("parent_text") or child_text,
                score=score,
                matched_text=child_text,
                source=payload.get("source", ""),
            )
        )
        if len(results) == top_k:
            break
    return results


class HybridSearcher:
    def __init__(self, qdrant_client: AsyncQdrantClient, embedder: EmbedderService):
        self.qdrant = qdrant_client
        self.embedder = embedder

    async def search(
        self,
        query: str,
        collection: str,
        top_k: int = 6,
        mode: SearchMode = "hybrid",
    ) -> list[SearchResult]:
        dense_vec, sparse_vec = await self.embedder.embed_query(query)
        # Children are deduplicated into parents afterwards, so fetch extra.
        limit = top_k * 3

        ranked_lists: list[list[str]] = []
        payloads: dict[str, dict] = {}

        if mode in ("hybrid", "dense"):
            dense_results = await self.qdrant.search(
                collection_name=collection,
                query_vector=NamedVector(name="dense", vector=dense_vec),
                limit=limit,
                with_payload=True,
            )
            ranked_lists.append([str(p.id) for p in dense_results])
            payloads.update({str(p.id): p.payload or {} for p in dense_results})

        if mode in ("hybrid", "sparse"):
            sparse_results = await self.qdrant.search(
                collection_name=collection,
                query_vector=NamedSparseVector(
                    name="sparse",
                    vector=SparseVector(
                        indices=sparse_vec.indices, values=sparse_vec.values
                    ),
                ),
                limit=limit,
                with_payload=True,
            )
            ranked_lists.append([str(p.id) for p in sparse_results])
            payloads.update({str(p.id): p.payload or {} for p in sparse_results})

        fused = rrf_fuse(ranked_lists)
        return collapse_to_parents(fused, payloads, top_k)
