# Enterprise RAG

> Self-hosted hybrid-search RAG — BGE-M3 dense + learned-sparse vectors · hand-written RRF fusion · parent-child retrieval · streaming FastAPI · Prometheus observability · end-to-end evaluation

[![CI](https://github.com/aarushi28902/enterprise-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/aarushi28902/enterprise-rag/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()

## Architecture

```mermaid
graph TB
    subgraph Ingestion
        PDF[PDF upload / text files] --> Chunker[Parent-Child Chunker\n1024-char parents / 256-char children]
        Chunker --> Embedder[BGE-M3, batched per parent\ndense + learned-sparse vectors]
        Embedder --> Qdrant[(Qdrant\nchild vectors + parent text in payload)]
    end

    subgraph Query
        Q[User Query] --> E2[BGE-M3 Embed]
        E2 --> HS[Dense search + sparse search\n18 child candidates each]
        HS --> Qdrant
        Qdrant --> RRF[RRF fusion, k = 60]
        RRF --> Parents[Collapse to unique parents\ntop 6]
        Parents --> Gen[Ollama llama3.2\nStreaming]
        Gen --> SSE[SSE Stream]
    end

    subgraph Observability
        API --> Prometheus[Prometheus]
        Prometheus --> Grafana[Grafana]
    end
```

## Key Design Decisions

| Decision | Alternative | Why |
|---|---|---|
| BGE-M3 dense + learned-sparse | Dense only, or BM25 | Exact terms and meaning from one model in one pass, free and local |
| Parent-child chunking | Fixed-size chunks | Search small children for precise matches; give the LLM the larger parent for context |
| Parent text in child payloads | Separate parent store | No second lookup at query time; each parent is copied into its ~5 children, fine at this scale |
| Hand-written RRF | Qdrant server-side fusion | Full control over k, per-retriever weighting and debugging; a few lines, unit-tested |
| Ollama local LLM | Hosted API | Zero cost, no data leaves the machine; slower and weaker than a frontier model |
| End-to-end eval, fact-level answer key | One hand-picked gold passage per question | Measures what the pipeline actually retrieves; credits any passage that states the answer |

## Evaluation

`eval/run_pipeline_eval.py` sends each of the 20 questions in `eval/datasets/taylor_swift_qa.jsonl` through the real retriever, in hybrid, dense-only and sparse-only modes, and checks the top 6 passages against an answer key.

**The answer key** lists, for each question, the facts the answer needs (26 facts in total) and every sentence in the corpus that states each fact (56 sentences, all copied verbatim and checked by a unit test). A question counts as **answered** only when every one of its facts appears in a retrieved passage. A passage contains a sentence when it holds at least 90% of the sentence's words.

| Mode | Answered @6 | MRR | Facts found |
|---|---|---|---|
| Hybrid (RRF) | **85%** (17/20) | 0.48 | 88% |
| Dense only | 80% (16/20) | 0.50 | 85% |
| Sparse only | 80% (16/20) | 0.53 | 83% |

**What this shows**

- On 20 questions the three modes are within one question of each other, so none is clearly better. Hybrid answers the most questions; sparse ranks its answers highest.
- Three questions fail in every mode. "Why are Folklore and Evermore called sister albums?" never retrieves the passage that says so; "What genre shift did Folklore represent?" never retrieves its answer sentence; "How was Folklore announced?" finds the release date but not the announcement. These are retrieval problems, not answer-key problems, and a reranker over a wider candidate set is the next thing to try.
- An earlier version of this eval matched one hand-picked passage per question. It scored hybrid at 50% and dense at 60%, but reviewing the misses showed the key was wrong: the Wikipedia pages repeat facts across passages, and two gold passages didn't contain their answer at all. Rebuilding the key changed the conclusion.

Per-question results (retrieved passages, rank, facts found) and a dated summary are in `eval/results/`. After changing only the answer key, `python -m eval.rescore` rescores the saved passages without Qdrant or the embedding model. With `--ragas`, `run_pipeline_eval` also generates answers with the real generator and scores RAGAS context precision and recall; `--full` adds faithfulness and answer relevancy, which need several LLM calls per question and are slow on CPU.

## Quick Start

```bash
# start Qdrant
./qdrant &

# start API
uvicorn src.rag.api.main:app --host 0.0.0.0 --port 8000 --reload

# ingest a document
curl -X POST http://localhost:8000/api/v1/ingest/pdf \
  -F "file=@your_document.pdf"

# query with streaming
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "your question here"}' \
  --no-buffer
```

### Reproduce the evaluation

```bash
python -m scripts.ingest_taylor_swift      # recreates the collection from data/
python -m eval.run_pipeline_eval           # retrieval metrics, all three modes
python -m eval.run_pipeline_eval --modes hybrid --ragas   # + RAGAS context metrics
python -m eval.rescore                     # rescore saved results after editing the answer key
pytest tests/                              # unit + API tests, no services needed
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/v1/ingest/pdf` | Ingest PDF document |
| POST | `/api/v1/query/stream` | Stream RAG response |
| GET | `/metrics` | Prometheus metrics |

## Metrics

- `rag_request_duration_seconds` — end-to-end latency histogram
- `rag_retrieval_duration_seconds` — Qdrant search latency
- `rag_generation_duration_seconds` — LLM generation latency
- `rag_docs_ingested_total` — total documents ingested
- `rag_active_requests` — currently active requests

## What I'd Do With More Time

- Replace CPU-based BGE-M3 with GPU inference for <100ms embedding latency
- Add cross-encoder reranking (ms-marco-MiniLM) between retrieval and generation
- Add a relevance threshold so weak retrieval returns "I don't know" instead of guessing
- Split on sentence boundaries and measure chunks in tokens rather than characters
- Move parent text to a separate store if the corpus grows large
- Add multi-tenancy with collection-per-tenant isolation
- Run the end-to-end eval in CI against a seeded Qdrant, with a GPU runner for faithfulness

## Stack

Python 3.11 · FastAPI · Qdrant · BGE-M3 · Ollama · Prometheus · RAGAS · pytest
