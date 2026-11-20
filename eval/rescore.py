"""Rescore saved retrieval results against the current answer key.

The passages each configuration retrieved are saved in eval/results/<config>.jsonl.
When only the answer key or the scoring changes, rescoring those saved passages
gives the same answered rate, MRR and fact coverage as a full rerun, without
Qdrant or the embedding model. Candidate recall is not recomputed (the 30
candidates are not saved), so rerun the full eval before quoting it.

    python -m eval.rescore
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from eval.retrieval_metrics import summarise
from eval.run_pipeline_eval import RESULTS_DIR, corpus_parents, load_rows, print_table, score


def main(dataset: str = "eval/datasets/taylor_swift_qa.jsonl", data_dir: str = "data") -> dict:
    rows = load_rows(Path(dataset))
    by_question = {r["question"]: r for r in rows}
    parents = corpus_parents(Path(data_dir))

    summary_path = RESULTS_DIR / "summary.json"
    old = json.loads(summary_path.read_text(encoding="utf-8"))
    # Keep everything rescoring can't recompute (latency, candidate recall,
    # abstention), and replace the scores that depend on the answer key.
    summary = {**old, "rescored": date.today().isoformat(), "modes": {}}

    for mode in old["modes"]:
        path = RESULTS_DIR / f"{mode}.jsonl"
        records = load_rows(path)
        for rec in records:
            rank, found, needed = score(by_question[rec["question"]], rec["contexts"], parents)
            rec.pop("first_hit_rank", None)
            rec.update(answer_rank=rank, facts_found=found, facts_needed=needed)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        summary["modes"][mode] = {
            **old["modes"][mode],
            **summarise(
                [r["answer_rank"] for r in records],
                [r["facts_found"] / r["facts_needed"] for r in records],
            ),
        }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_table(summary)
    return summary


if __name__ == "__main__":
    main()
