"""Retrieval metrics that need no LLM: did the retrieved passages contain the answer?

Each eval question lists the facts its answer needs, and for each fact every
corpus sentence that states it (the answer key in eval/datasets). A question is
answered when, for every fact, at least one retrieved passage contains one of
that fact's sentences.

"Contains" is by word overlap rather than exact text, so formatting and
punctuation don't matter: a passage contains a sentence when it holds at least
HIT_THRESHOLD of the sentence's words. A sentence cut in two by a chunk
boundary can never fully appear in one passage, so its bar drops to the best
overlap any passage in the corpus achieves.
"""
from __future__ import annotations

import re

HIT_THRESHOLD = 0.9


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def overlap(gold: str, passage: str) -> float:
    """Fraction of the gold sentence's words that appear in the passage."""
    g = words(gold)
    return len(g & words(passage)) / len(g) if g else 0.0


def hit_bar(gold: str, corpus_passages: list[str]) -> float:
    """The overlap a retrieved passage needs to count as containing the gold."""
    best = max((overlap(gold, p) for p in corpus_passages), default=0.0)
    return min(HIT_THRESHOLD, best)


def first_hit_rank(gold_passages: list[str], retrieved: list[str], bars: list[float]) -> int | None:
    """1-based rank of the first retrieved passage containing any gold sentence."""
    for rank, passage in enumerate(retrieved, start=1):
        for gold, bar in zip(gold_passages, bars):
            if bar > 0 and overlap(gold, passage) >= bar:
                return rank
    return None


def answer_rank(
    facts: list[list[str]], retrieved: list[str], bars: list[list[float]]
) -> tuple[int | None, int]:
    """Score one question against its answer key.

    Returns (rank, facts_found): rank is the position by which every fact has
    appeared (None if any fact is missing), facts_found counts facts present.
    """
    fact_ranks = [first_hit_rank(golds, retrieved, b) for golds, b in zip(facts, bars)]
    found = sum(r is not None for r in fact_ranks)
    if found < len(facts):
        return None, found
    return max(fact_ranks), found


def summarise(ranks: list[int | None], fact_coverage: list[float] | None = None) -> dict:
    n = len(ranks)
    if n == 0:
        return {"questions": 0, "answered": 0.0, "mrr": 0.0}
    hits = [r for r in ranks if r is not None]
    out = {
        "questions": n,
        "answered": round(len(hits) / n, 3),
        "mrr": round(sum(1 / r for r in hits) / n, 3),
    }
    if fact_coverage is not None:
        out["fact_coverage"] = round(sum(fact_coverage) / n, 3)
    return out
