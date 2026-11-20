from types import SimpleNamespace

import pytest

from eval.retrieval_metrics import first_hit_rank, hit_bar, overlap, summarise
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
    # Paraphrased gold that no passage fully contains: bar drops to the best match.
    bar = hit_bar("Evermore is the ninth studio album by Swift", CORPUS)
    assert bar == pytest.approx(6 / 8)


def test_first_hit_rank_finds_the_right_position():
    gold = ["Betty is a country and folk rock song"]
    bars = [hit_bar(g, CORPUS) for g in gold]
    assert first_hit_rank(gold, [CORPUS[0], CORPUS[2]], bars) == 2
    assert first_hit_rank(gold, [CORPUS[0], CORPUS[1]], bars) is None


def test_summarise_hit_rate_and_mrr():
    s = summarise([1, 2, None, None])
    assert s["hit_rate"] == 0.5
    assert s["mrr"] == pytest.approx((1 + 0.5) / 4, abs=1e-3)
    assert summarise([])["questions"] == 0


class FakeSearcher:
    """Returns fixed passages so we can check the eval scores what search returns."""

    def __init__(self, passages):
        self.passages = passages
        self.calls = []

    async def search(self, query, collection, top_k, mode):
        self.calls.append((query, mode))
        return [SimpleNamespace(text=t, source="doc") for t in self.passages[:top_k]]


@pytest.mark.asyncio
async def test_run_mode_scores_retrieved_text_not_dataset_contexts():
    rows = [{
        "question": "When was Evermore released?",
        "ground_truth": "December 2020",
        "contexts": ["Evermore is the ninth studio album, released in December 2020."],
    }]
    searcher = FakeSearcher([CORPUS[0], CORPUS[1]])
    records = await run_mode("dense", rows, searcher, "col", 6, CORPUS)

    assert searcher.calls == [("When was Evermore released?", "dense")]
    assert records[0]["contexts"] == [CORPUS[0], CORPUS[1]]  # what search returned
    assert records[0]["first_hit_rank"] == 2

    missing = await run_mode("dense", rows, FakeSearcher([CORPUS[2]]), "col", 6, CORPUS)
    assert missing[0]["first_hit_rank"] is None
