"""Ingest the Taylor Swift Wikipedia corpus into Qdrant (recreates the collection)."""
import asyncio
from pathlib import Path

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, SparseVectorParams, VectorParams

from src.rag.ingestion.chunker import ParentChildChunker
from src.rag.ingestion.pipeline import build_points
from src.rag.retrieval.embedder import EmbedderService

COLLECTION = "enterprise_rag"
FILES = [
    ("data/folklore_album.txt", "Folklore Album - Wikipedia"),
    ("data/evermore_album.txt", "Evermore Album - Wikipedia"),
    ("data/betty_song.txt", "Betty Song - Wikipedia"),
    ("data/cardigan_song.txt", "Cardigan Song - Wikipedia"),
    ("data/willow_song.txt", "Willow Song - Wikipedia"),
]


async def main():
    client = AsyncQdrantClient(url="http://localhost:6333")
    embedder = EmbedderService()

    print("Loading embedder...")
    await embedder.load()

    # Start clean so no points from older chunking (without parent_text) remain.
    if await client.collection_exists(COLLECTION):
        print("Deleting existing collection...")
        await client.delete_collection(COLLECTION)

    print("Creating collection...")
    await client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )

    chunker = ParentChildChunker(parent_size=1024, child_size=256, overlap=32)

    for filepath, source in FILES:
        text = Path(filepath).read_text(encoding="utf-8")
        print(f"\nIngesting: {source} ({len(text)} chars)")

        pairs = list(chunker.chunk(text, metadata={"source": source}))
        points = await build_points(pairs, embedder, source)
        await client.upsert(collection_name=COLLECTION, points=points)
        print(f"  -> {len(pairs)} parents, {len(points)} children ingested")

    count = await client.count(collection_name=COLLECTION)
    print(f"\nTotal points in collection: {count.count}")


if __name__ == "__main__":
    asyncio.run(main())
