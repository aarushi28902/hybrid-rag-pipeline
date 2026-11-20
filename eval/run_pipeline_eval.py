"""End-to-end evaluation: every question goes through the real retriever (and,
with --ragas, the real generator). Nothing is read from hand-written answers.

Usage (Qdrant running, corpus ingested with scripts/ingest_taylor_swift.py):

    python -m eval.run_pipeline_eval                  # retrieval metrics, all 3 modes
    python -m eval.run_pipeline_eval --modes hybrid --ragas          # + RAGAS context metrics
    python -m eval.run_pipeline_eval --modes hybrid --ragas --full   # + faithfulness, relevancy

Retrieval metrics need no LLM:
  hit_rate  share of questions where a top-k passage contains the gold passage
  mrr       mean of 1 / rank of the first such passage (0 when none)

Writes per-question rows to eval/results/<mode>.jsonl and a summary to
eval/results/summary.json, so every number in the README can be traced.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from eval.retrieval_metrics import first_hit_rank, hit_bar, summarise
from src.rag.ingestion.chunker import ParentChildChunker
from src.rag.retrieval.embedder import EmbedderService
from src.rag.retrieval.hybrid_search import HybridSearcher

RESULTS_DIR = Path("eval/results")
# Must match the chunker settings used by scripts/ingest_taylor_swift.py.
CHUNKER = ParentChildChunker(parent_size=1024, child_size=256, overlap=32)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def corpus_parents(data_dir: Path) -> list[str]:
    parents: list[str] = []
    for f in sorted(data_dir.glob("*.txt")):
        parents += [p.text for p, _ in CHUNKER.chunk(f.read_text(encoding="utf-8"), {"source": f.name})]
    return parents


async def generate(generator, question: str, results) -> str:
    return "".join([tok async for tok in generator.stream(query=question, chunks=results)])


async def run_mode(mode, rows, searcher, collection, top_k, parents, generator=None):
    out = []
    for i, row in enumerate(rows, start=1):
        results = await searcher.search(row["question"], collection, top_k=top_k, mode=mode)
        retrieved = [r.text for r in results]
        gold = row["contexts"]  # hand-picked gold passages, used only as the answer key
        bars = [hit_bar(g, parents) for g in gold]
        record = {
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "retrieved_sources": [r.source for r in results],
            "contexts": retrieved,
            "first_hit_rank": first_hit_rank(gold, retrieved, bars),
        }
        if generator is not None:
            record["answer"] = await generate(generator, row["question"], results)
        out.append(record)
        print(f"  [{mode}] {i}/{len(rows)} hit@rank={record['first_hit_rank']}", flush=True)
    return out


def run_ragas(records: list[dict], full: bool) -> dict:
    import numpy as np
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    from eval.ragas_config import get_ragas_embeddings, get_ragas_llm

    metrics = [context_precision, context_recall] + ([faithfulness, answer_relevancy] if full else [])
    llm, embeddings = get_ragas_llm(), get_ragas_embeddings()
    for m in metrics:
        m.llm = llm
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings

    ds = Dataset.from_list(
        [{k: r[k] for k in ("question", "answer", "contexts", "ground_truth")} for r in records]
    )
    df = evaluate(ds, metrics=metrics, raise_exceptions=False).to_pandas()
    scores = {}
    for m in metrics:
        if m.name in df.columns:
            vals = df[m.name].dropna()
            scores[m.name] = round(float(np.nanmean(vals)), 3) if len(vals) else None
            scores[f"{m.name}_scored"] = int(len(vals))  # rows that didn't time out
    return scores


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="eval/datasets/taylor_swift_qa.jsonl")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--collection", default="enterprise_rag")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--modes", nargs="+", default=["hybrid", "dense", "sparse"], choices=["hybrid", "dense", "sparse"])
    p.add_argument("--ragas", action="store_true", help="generate answers and score context precision/recall")
    p.add_argument("--full", action="store_true", help="with --ragas, also faithfulness and answer relevancy (slow on CPU)")
    args = p.parse_args()

    rows = load_rows(Path(args.dataset))
    parents = corpus_parents(Path(args.data_dir))
    print(f"{len(rows)} questions, {len(parents)} corpus passages, top_k={args.top_k}")

    embedder = EmbedderService()
    print("Loading embedder...")
    await embedder.load()
    searcher = HybridSearcher(qdrant_client=AsyncQdrantClient(url=args.qdrant_url), embedder=embedder)

    generator = None
    if args.ragas:
        from src.rag.generation.generator import StreamingGenerator
        generator = StreamingGenerator()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "date": date.today().isoformat(),
        "dataset": args.dataset,
        "top_k": args.top_k,
        "hit_threshold": "passage holds >= 90% of the gold passage's words (or the corpus best)",
        "modes": {},
    }

    for mode in args.modes:
        records = await run_mode(
            mode, rows, searcher, args.collection, args.top_k, parents,
            generator if (args.ragas and mode == args.modes[0]) else None,
        )
        with (RESULTS_DIR / f"{mode}.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary["modes"][mode] = summarise([r["first_hit_rank"] for r in records])
        if args.ragas and mode == args.modes[0]:
            print(f"Running RAGAS on {mode} (slow on CPU)...")
            summary["modes"][mode]["ragas"] = run_ragas(records, args.full)

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'mode':<8} {'hit@' + str(args.top_k):>8} {'MRR':>6}")
    for mode, s in summary["modes"].items():
        print(f"{mode:<8} {s['hit_rate']:>8.3f} {s['mrr']:>6.3f}")
        if "ragas" in s:
            print("         ragas:", s["ragas"])
    print(f"\nSaved per-question rows and summary to {RESULTS_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
