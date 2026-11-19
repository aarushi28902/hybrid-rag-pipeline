from __future__ import annotations

import io

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, SparseVector

from src.rag.config import get_settings
from src.rag.ingestion.chunker import Chunk, ParentChildChunker
from src.rag.retrieval.embedder import EmbedderService

logger = structlog.get_logger()


async def build_points(
    pairs: list[tuple[Chunk, list[Chunk]]],
    embedder: EmbedderService,
    source: str,
) -> list[PointStruct]:
    """Embed every child chunk and attach its parent's text to the payload.

    Only children are embedded and searched (small chunks match precisely).
    Each child carries its parent's text, so retrieval can hand the LLM the
    larger parent for context without a second lookup.
    """
    points: list[PointStruct] = []
    for parent, children in pairs:
        vectors = await embedder.embed_documents([c.text for c in children])
        for child, (dense, sparse) in zip(children, vectors):
            points.append(
                PointStruct(
                    id=child.chunk_id,
                    vector={
                        "dense": dense,
                        "sparse": SparseVector(
                            indices=sparse.indices,
                            values=sparse.values,
                        ),
                    },
                    payload={
                        "text": child.text,
                        "chunk_id": child.chunk_id,
                        "parent_id": parent.chunk_id,
                        "parent_text": parent.text,
                        "source": source,
                    },
                )
            )
    return points


class IngestionPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.chunker = ParentChildChunker()
        self.embedder = EmbedderService()

    async def run_pdf(self, content: bytes, filename: str, collection: str) -> None:
        from pypdf import PdfReader

        logger.info("ingestion_start", filename=filename)

        reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        await self._ingest_text(text=text, source=filename, collection=collection)
        logger.info("ingestion_complete", filename=filename)

    async def _ingest_text(self, text: str, source: str, collection: str) -> None:
        client = AsyncQdrantClient(url=self.settings.qdrant_url)
        await self.embedder.load()

        pairs = list(self.chunker.chunk(text, metadata={"source": source}))
        points = await build_points(pairs, self.embedder, source)

        await client.upsert(collection_name=collection, points=points)
        logger.info("upsert_complete", n_parents=len(pairs), n_points=len(points))
