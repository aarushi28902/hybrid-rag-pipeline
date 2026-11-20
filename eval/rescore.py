"""Rescore saved retrieval results against the current answer key.

The passages each mode retrieved are saved in eval/results/<mode>.jsonl. When
only the answer key or the scoring changes, rescoring those saved passages gives
the same numbers as a full rerun, without Qdrant or the embedding model.

    python -m eval.rescore
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from eval.retrieval_metrics import summarise
from eval.run_pipeline_eval import RESULTS_DIR, corpus_parents, load_rows, score


def main(dataset: str = "eval/datasets/taylor_swift_qa.jsonl", data_dir: str = "data") -> dict:
    rows = load_rows(Path(dataset))
    by_question = {r["question"]: r for r in rows}
    parents = corpus_parents(Path(data_dir))

    summary_path = RESULTS_DIR / "summary.json"
    old = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = {
        "date": old.get("date"),  # when retrieval ran
        "rescored": date.today().isoformat(),
        "dataset": dataset,
        "top_k": old.get("top_k"),
        "scoring": "answered = every fact in the answer key appears in a top-k passage; "
        "a passage contains a sentence when it holds >= 90% of its words (or the corpus best)",
        "modes": {},
    }

    for mode in old["modes"]:
        path = RESULTS_DIR / f"{mode}.jsonl"
        records = load_rows(path)
        for rec in records:
            rank, found, needed = score(by_question[rec["question"]], rec["contexts"], parents)
            rec.pop("first_hit_rank", None)
            rec.update(answer_rank=rank, facts_found=found, facts_needed=needed)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        summary["modes"][mode] = summarise(
            [r["answer_rank"] for r in records],
            [r["facts_found"] / r["facts_needed"] for r in records],
        )

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{'mode':<8} {'answered@' + str(summary['top_k']):>12} {'MRR':>6} {'facts':>6}")
    for mode, s in summary["modes"].items():
        print(f"{mode:<8} {s['answered']:>12.3f} {s['mrr']:>6.3f} {s['fact_coverage']:>6.3f}")
    return summary


if __name__ == "__main__":
    main()
