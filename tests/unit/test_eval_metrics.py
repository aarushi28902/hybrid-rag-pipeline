import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.retrieval_metrics import answer_rank, first_hit_rank, hit_bar, overlap, summarise
from eval.run_pipeline_eval import run_mode

CORPUS = [
    "Folklore was released on July 24, 2020, and announced the day before.",
    "Evermore is the ninth studio album, released in December 2020.",
    "Betty is a country and folk rock song about a teenage love triangle.",
]


def test_overlap_is_share_of_gold_words_found():
    assert overlap("Folklore was released", CORPUS[0]) == 1.0
    assert overlap("Folklore was released", "Folklore") == pytest.approx(1 / 3)
    assert overlap("", CORPUS[0]) == 0.0


def test_hit_bar_caps_at_threshold_or_corpus_best():
    assert hit_bar("Evermore is the ninth studio album", CORPUS) == 0.9
    # A sentence no passage fully contains (e.g. cut by a chunk boundary).
    bar = hit_bar("Evermore is the ninth studio album by Swift", CORPUS)
    assert bar == pytest.approx(6 / 8)


def test_first_hit_rank_finds_the_right_position():
    gold = ["Betty is a country and folk rock song"]
    bars = [hit_bar(g, CORPUS) for g in gold]
    assert first_hit_rank(gold, [CORPUS[0], CORPUS[2]], bars) == 2
    assert first_hit_rank(gold, [CORPUS[0], CORPUS[1]], bars) is None


def test_answer_rank_needs_every_fact():
    facts = [["Folklore was released on July 24, 2020"], ["Evermore is the ninth studio album"]]
    bars = [[hit_bar(s, CORPUS) for s in f] for f in facts]
    # Both facts present: answered at the rank where the last one appears.
    assert answer_rank(facts, [CORPUS[0], CORPUS[2], CORPUS[1]], bars) == (3, 2)
    # One fact missing: not answered, but partial credit is reported.
    assert answer_rank(facts, [CORPUS[0], CORPUS[2]], bars) == (None, 1)


def test_answer_rank_accepts_any_sentence_for_a_fact():
    facts = [["Evermore is the ninth studio album", "Betty is a country and folk rock song"]]
    bars = [[hit_bar(s, CORPUS) for s in f] for f in facts]
    assert answer_rank(facts, [CORPUS[0], CORPUS[2]], bars) == (2, 1)


def test_summarise_answered_mrr_and_coverage():
    s = summarise([1, 2, None, None], [1.0, 1.0, 0.5, 0.0])
    assert s["answered"] == 0.5
    assert s["mrr"] == pytest.approx((1 + 0.5) / 4, abs=1e-3)
    assert s["fact_coverage"] == 0.625
    assert summarise([])["questions"] == 0


def test_answer_key_sentences_all_come_from_the_corpus():
    """Every evidence sentence must appear in data/ (guards against typos)."""
    import re

    def norm(t):
        return re.sub(r"\s+", " ", t)

    corpus = norm(" ".join(p.read_text(encoding="utf-8") for p in Path("data").glob("*.txt")))
    rows = [json.loads(l) for l in Path("eval/datasets/taylor_swift_qa.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 20
    for row in rows:
        assert row["evidence"], row["question"]
        for fact in row["evidence"]:
            assert fact, row["question"]
            for sentence in fact:
                assert norm(sentence) in corpus, sentence[:80]


class FakeSearcher:
    """Returns fixed passages so we can check the eval scores what search returns."""

    def __init__(self, passages):
        self.passages = passages
        self.calls = []

    async def search(self, query, collection, top_k, mode):
        self.calls.append((query, mode))
        return [SimpleNamespace(text=t, source="doc") for t in self.passages[:top_k]]


@pytest.mark.asyncio
async def test_run_mode_scores_retrieved_text():
    rows = [{
        "question": "When was Evermore released?",
        "ground_truth": "December 2020",
        "evidence": [["Evermore is the ninth studio album, released in December 2020."]],
    }]
    searcher = FakeSearcher([CORPUS[0], CORPUS[1]])
    records = await run_mode("dense", rows, searcher, "col", 6, CORPUS)

    assert searcher.calls == [("When was Evermore released?", "dense")]
    assert records[0]["contexts"] == [CORPUS[0], CORPUS[1]]  # what search returned
    assert records[0]["answer_rank"] == 2
    assert (records[0]["facts_found"], records[0]["facts_needed"]) == (1, 1)

    missing = await run_mode("dense", rows, FakeSearcher([CORPUS[2]]), "col", 6, CORPUS)
    assert missing[0]["answer_rank"] is None
    assert missing[0]["facts_found"] == 0
