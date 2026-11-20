"""End-to-end evaluation and reranking ablation.

Every question goes through the real retriever (and, with --ragas, the real
generator). Nothing is read from hand-written answers.

Usage (Qdrant running, corpus ingested with scripts/ingest_taylor_swift.py):

    python -m eval.run_pipeline_eval                      # all five configurations
    python -m eval.run_pipeline_eval --configs hybrid dense    # a subset
    python -m eval.run_pipeline_eval --configs hybrid_rerank --ragas   # + RAGAS

Configurations
  hybrid, dense, sparse   retrieval only, top 6
  hybrid_rerank           hybrid top 30 candidates, cross-encoder rerank, top 6
  dense_rerank            same with dense retrieval

Metrics (no LLM needed). The answer key (eval/datasets) lists, for each
question, the facts its answer needs and every corpus sentence stating each fact.
  answered        share of questions where the top 6 passages contain every fact
  mrr             mean of 1 / (rank by which every fact has appeared), 0 when not answered
  fact_coverage   mean share of a question's facts found (partial credit)
  candidate_recall (rerank configs) share answered within the 30 candidates:
                  the ceiling a reranker can reach, since it only reorders
  retrieval_ms / rerank_ms   mean latency per question on this machine

Abstention (with hybrid_rerank): the best reranked passage's score is recorded
for the 50 answerable questions and for 10 questions the corpus can't answer
(eval/datasets/unanswerable.jsonl). The recommended threshold is the highest
one that wrongly refuses at most MAX_FALSE_ABSTAIN of answerable questions.

Writes per-question rows to eval/results/<config>.jsonl and a summary to
eval/results/summary.json, so every number in the README can be traced.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import date
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from eval.retrieval_metrics import answer_rank, hit_bar, summarise
from src.rag.ingestion.chunker import ParentChildChunker
from src.rag.retrieval.embedder import EmbedderService
from src.rag.retrieval.hybrid_search import HybridSearcher

RESULTS_DIR = Path("eval/results")
# Must match the chunker settings used by scripts/ingest_taylor_swift.py.
CHUNKER = ParentChildChunker(parent_size=1024, child_size=256, overlap=32)
CONFIGS = {
    "hybrid": ("hybrid", False),
    "dense": ("dense", False),
    "sparse": ("sparse", False),
    "hybrid_rerank": ("hybrid", True),
    "dense_rerank": ("dense", True),
}
MAX_FALSE_ABSTAIN = 0.04  # at most 2 of 50 answerable questions refused


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def corpus_parents(data_dir: Path) -> list[str]:
    parents: list[str] = []
    for f in sorted(data_dir.glob("*.txt")):
        parents += [p.text for p, _ in CHUNKER.chunk(f.read_text(encoding="utf-8"), {"source": f.name})]
    return parents


def score(row: dict, retrieved: list[str], parents: list[str]) -> tuple[int | None, int, int]:
    """(answer rank or None, facts found, facts needed) for one question."""
    facts = row["evidence"]  # [[sentences stating fact 1], [fact 2], ...]
    bars = [[hit_bar(s, parents) for s in fact] for fact in facts]
    rank, found = answer_rank(facts, retrieved, bars)
    return rank, found, len(facts)


async def generate(generator, question: str, results) -> str:
    return "".join([tok async for tok in generator.stream(query=question, chunks=results)])


async def retrieve(question, searcher, collection, mode, rerank, reranker, top_k, candidates):
    """Run one configuration; returns (results, candidates or None, retrieval s, rerank s)."""
    t = time.perf_counter()
    if not rerank:
        results = await searcher.search(question, collection, top_k=top_k, mode=mode)
        return results, None, time.perf_counter() - t, 0.0
    pool = await searcher.search(question, collection, top_k=candidates, mode=mode)
    t_ret = time.perf_counter() - t
    t = time.perf_counter()
    results = await reranker.rerank(question, pool, top_k=top_k)
    return results, pool, t_ret, time.perf_counter() - t


async def run_config(name, rows, searcher, reranker, collection, top_k, candidates, parents, generator=None):
    mode, rerank = CONFIGS[name]
    out = []
    for i, row in enumerate(rows, start=1):
        results, pool, t_ret, t_rr = await retrieve(
            row["question"], searcher, collection, mode, rerank, reranker, top_k, candidates
        )
        retrieved = [r.text for r in results]
        rank, found, needed = score(row, retrieved, parents)
        record = {
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "retrieved_sources": [r.source for r in results],
            "contexts": retrieved,
            "answer_rank": rank,
            "facts_found": found,
            "facts_needed": needed,
            "retrieval_ms": round(t_ret * 1000, 1),
            "rerank_ms": round(t_rr * 1000, 1),
        }
        if rerank:
            record["rerank_scores"] = [round(r.rerank_score, 4) for r in results]
            record["candidate_answer_rank"] = score(row, [p.text for p in pool], parents)[0]
        if generator is not None:
            record["answer"] = await generate(generator, row["question"], results)
        out.append(record)
        print(f"  [{name}] {i}/{len(rows)} facts {found}/{needed}, answered@rank={rank}", flush=True)
    return out


def config_summary(records: list[dict], rerank: bool) -> dict:
    n = len(records)
    s = summarise(
        [r["answer_rank"] for r in records],
        [r["facts_found"] / r["facts_needed"] for r in records],
    )
    s["retrieval_ms"] = round(sum(r["retrieval_ms"] for r in records) / n, 1)
    if rerank:
        s["rerank_ms"] = round(sum(r["rerank_ms"] for r in records) / n, 1)
        s["candidate_recall"] = round(sum(r["candidate_answer_rank"] is not None for r in records) / n, 3)
    return s


def choose_threshold(answerable: list[float], unanswerable: list[float], max_false: float = MAX_FALSE_ABSTAIN) -> dict:
    """Threshold that refuses the most unanswerable questions while refusing at
    most max_false of answerable ones.

    A question is refused when its best reranked passage scores below the
    threshold. Candidates are midpoints between neighbouring observed scores, so
    the threshold never sits exactly on a real question's score; on ties the
    lowest threshold wins, leaving the most margin for answerable questions.
    """
    scores = sorted(set(answerable + unanswerable))
    candidates = [0.0] + [(a + b) / 2 for a, b in zip(scores, scores[1:])]
    best = {"threshold": 0.0, "false_abstain": 0.0, "correct_abstain": 0.0}
    for t in candidates:
        false_rate = sum(s < t for s in answerable) / len(answerable)
        if false_rate > max_false:
            break
        correct = sum(s < t for s in unanswerable) / len(unanswerable)
        if correct > best["correct_abstain"]:
            best = {"threshold": round(t, 4), "false_abstain": round(false_rate, 3), "correct_abstain": round(correct, 3)}
    return best


async def abstention_study(answerable_records, unanswerable_rows, searcher, reranker, collection, top_k, candidates):
    unans = []
    for row in unanswerable_rows:
        results, _, _, _ = await retrieve(row["question"], searcher, collection, "hybrid", True, reranker, top_k, candidates)
        unans.append({"question": row["question"], "top_score": round(results[0].rerank_score, 4) if results else 0.0})
    ans_scores = [r["rerank_scores"][0] if r["rerank_scores"] else 0.0 for r in answerable_records]
    chosen = choose_threshold(ans_scores, [u["top_score"] for u in unans])
    return {
        "config": "hybrid_rerank",
        "max_false_abstain": MAX_FALSE_ABSTAIN,
        **chosen,
        "answerable_top_scores": {"min": min(ans_scores), "median": sorted(ans_scores)[len(ans_scores) // 2]},
        "unanswerable": unans,
    }


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


def print_table(summary: dict) -> None:
    print(f"\n{'config':<14} {'answered':>9} {'MRR':>6} {'facts':>6} {'cand@30':>8} {'ret ms':>7} {'rr ms':>7}")
    for name, s in summary["modes"].items():
        cand = f"{s['candidate_recall']:.3f}" if "candidate_recall" in s else "-"
        rr = f"{s['rerank_ms']:.0f}" if "rerank_ms" in s else "-"
        print(f"{name:<14} {s['answered']:>9.3f} {s['mrr']:>6.3f} {s['fact_coverage']:>6.3f} {cand:>8} "
              f"{s.get('retrieval_ms', 0):>7.0f} {rr:>7}")
    a = summary.get("abstention")
    if a:
        print(f"\nAbstention threshold {a['threshold']}: refuses {a['correct_abstain']:.0%} of unanswerable, "
              f"{a['false_abstain']:.0%} of answerable questions.")
        print(f"Set it with ABSTAIN_THRESHOLD={a['threshold']} in .env")


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="eval/datasets/taylor_swift_qa.jsonl")
    p.add_argument("--unanswerable", default="eval/datasets/unanswerable.jsonl")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--collection", default="enterprise_rag")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--candidates", type=int, default=30, help="pool size the reranker chooses from")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
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

    reranker = None
    if any(CONFIGS[c][1] for c in args.configs):
        from src.rag.retrieval.reranker import Reranker
        print("Loading reranker...")
        reranker = Reranker()
        await reranker.load()

    generator = None
    if args.ragas:
        from src.rag.generation.generator import StreamingGenerator
        generator = StreamingGenerator()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "date": date.today().isoformat(),
        "dataset": args.dataset,
        "questions": len(rows),
        "top_k": args.top_k,
        "candidates": args.candidates,
        "scoring": "answered = every fact in the answer key appears in a top-k passage; "
        "a passage contains a sentence when it holds >= 90% of its words (or the corpus best)",
        "modes": {},
    }

    for name in args.configs:
        records = await run_config(
            name, rows, searcher, reranker, args.collection, args.top_k, args.candidates, parents,
            generator if (args.ragas and name == args.configs[0]) else None,
        )
        with (RESULTS_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary["modes"][name] = config_summary(records, CONFIGS[name][1])
        if args.ragas and name == args.configs[0]:
            print(f"Running RAGAS on {name} (slow on CPU)...")
            summary["modes"][name]["ragas"] = run_ragas(records, args.full)
        if name == "hybrid_rerank" and Path(args.unanswerable).exists():
            print("Abstention study on unanswerable questions...")
            summary["abstention"] = await abstention_study(
                records, load_rows(Path(args.unanswerable)), searcher, reranker,
                args.collection, args.top_k, args.candidates,
            )

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_table(summary)
    print(f"\nSaved per-question rows and summary to {RESULTS_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
