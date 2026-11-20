"""Cross-encoder reranking.

Hybrid search compares the query and each passage as two separate vectors.
A cross-encoder reads the query and the passage together in one forward pass,
so it judges relevance much more accurately, but it is too slow to run over a
whole corpus. So it runs second: over the few dozen candidates retrieval found.

Its score is also what makes abstention possible. RRF scores depend only on
rank (the top result always scores the same, relevant or not), so they can't
say "nothing relevant was found". A cross-encoder score can.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import replace

from src.rag.retrieval.hybrid_search import SearchResult

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    async def load(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()

    def score_sync(self, query: str, passages: list[str]) -> list[float]:
        """Relevance of each passage to the query, as a probability in [0, 1]."""
        import torch

        if not passages:
            return []
        self._load_sync()
        logits: list[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            inputs = self._tokenizer(
                [query] * len(batch), batch,
                padding=True, truncation=True, max_length=512, return_tensors="pt",
            )
            with torch.no_grad():
                out = self._model(**inputs).logits.view(-1)
            logits += out.tolist()
        # The model outputs one logit per pair; a sigmoid maps it to [0, 1]
        # so a threshold reads as "how likely this passage is relevant".
        return [1 / (1 + math.exp(-x)) for x in logits]

    async def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Reorder retrieval results by cross-encoder score and keep the top_k."""
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, self.score_sync, query, [r.text for r in results])
        return rerank_by_scores(results, scores, top_k)


def rerank_by_scores(results: list[SearchResult], scores: list[float], top_k: int) -> list[SearchResult]:
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [replace(r, rerank_score=s) for r, s in ranked[:top_k]]


def should_abstain(results: list[SearchResult], threshold: float) -> bool:
    """Abstain when nothing was found or the best passage scores below the threshold."""
    if not results:
        return True
    best = results[0].rerank_score
    return best is not None and best < threshold
