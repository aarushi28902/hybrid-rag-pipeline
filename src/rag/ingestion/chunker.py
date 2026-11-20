from __future__ import annotations
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def deterministic_uuid(text: str) -> str:
    """Same text always produces the same UUID - enables deduplication."""
    hash_bytes = hashlib.sha256(text.encode()).digest()[:16]
    return str(uuid.UUID(bytes=hash_bytes))


class ParentChildChunker:
    def __init__(
        self,
        parent_size: int = 1024,
        child_size: int = 256,
        overlap: int = 32,
    ):
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap

    def _split(self, text: str, size: int) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start = end - self.overlap
        return chunks

    def chunk(
        self,
        text: str,
        metadata: dict,
    ) -> Iterator[tuple[Chunk, list[Chunk]]]:
        parent_texts = self._split(text, self.parent_size)
        source = str(metadata.get("source", ""))

        for position, parent_text in enumerate(parent_texts):
            # Deterministic, so re-ingesting a document overwrites its chunks
            # instead of creating duplicates with new random IDs.
            parent_id = deterministic_uuid(f"{source}|{position}|{parent_text}")

            parent = Chunk(
                chunk_id=parent_id,
                parent_id=parent_id,
                text=parent_text,
                metadata={**metadata, "level": "parent"},
            )

            child_texts = self._split(parent_text, self.child_size)
            children = [
                Chunk(
                    # Include the parent ID: the same child text inside two
                    # different parents must not overwrite each other.
                    chunk_id=deterministic_uuid(f"{parent_id}|{i}|{ct}"),
                    parent_id=parent_id,
                    text=ct,
                    metadata={**metadata, "level": "child"},
                )
                for i, ct in enumerate(child_texts)
            ]

            yield parent, children
