"""Retrieval metrics that need no LLM: does the retrieved text contain the gold passage?

Each eval question has a hand-picked gold passage from the corpus. Some are
lightly paraphrased, so matching is by word overlap rather than exact text:
a retrieved passage is a hit when it contains at least HIT_THRESHOLD of the
gold passage's words, or, if no passage in the corpus reaches that, as much
as the best passage in the corpus does.
"""
from __future__ import annotations

import re

HIT_THRESHOLD = 0.9


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def overlap(gold: str, passage: str) -> float:
    """Fraction of the gold passage's words that appear in the passage."""
    g = words(gold)
    return len(g & words(passage)) / len(g) if g else 0.0


def hit_bar(gold: str, corpus_passages: list[str]) -> float:
    """The overlap a retrieved passage needs to count as containing the gold."""
    best = max((overlap(gold, p) for p in corpus_passages), default=0.0)
    return min(HIT_THRESHOLD, best)


def first_hit_rank(gold_passages: list[str], retrieved: list[str], bars: list[float]) -> int | None:
    """1-based rank of the first retrieved passage containing any gold passage."""
    for rank, passage in enumerate(retrieved, start=1):
        for gold, bar in zip(gold_passages, bars):
            if bar > 0 and overlap(gold, passage) >= bar:
                return rank
    return None


def summarise(ranks: list[int | None]) -> dict:
    n = len(ranks)
    if n == 0:
        return {"questions": 0, "hit_rate": 0.0, "mrr": 0.0}
    hits = [r for r in ranks if r is not None]
    return {
        "questions": n,
        "hit_rate": round(len(hits) / n, 3),
        "mrr": round(sum(1 / r for r in hits) / n, 3),
    }
