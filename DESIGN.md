# RAG + Telemetry + Evals Design

## Goal

Build a deterministic, local-first retrieval system that demonstrates the complete AI-engineering loop: corpus ingestion, chunking, indexing, retrieval, privacy-preserving telemetry, offline evaluation, tests, and reproducible packaging.

## Contracts

- Corpus: JSON list of `{doc_id, title, text}`.
- Retriever: deterministic BM25-style ranking.
- Output: ranked chunks with stable document/chunk identifiers.
- Telemetry: JSONL events for index build and retrieval.
- Privacy boundary: retrieval telemetry persists query SHA-256 and query length, not raw query text.
- Evaluation: offline labeled queries with Recall@1, Recall@K, and MRR.

## Gate criteria

DESIGN requires documented data, retrieval, telemetry, evaluation, and evidence boundaries.
IMPLEMENT requires executable retrieval/telemetry/eval code and tests.
EVALUATE requires the same-run test suite plus retrieval metrics above declared thresholds.
PACKAGE requires runnable docs, demo, case study, manifests, and ZIP.
