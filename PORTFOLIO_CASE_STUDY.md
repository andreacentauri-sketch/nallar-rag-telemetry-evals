# Portfolio Case Study — RAG + Telemetry + Evals

## Problem

A retrieval system is not portfolio-ready if it only returns documents. It also needs measurable behavior, observability, privacy boundaries, and reproducibility.

## Solution

I implemented a local deterministic RAG retrieval layer using chunking and BM25-style ranking, added privacy-preserving JSONL telemetry, and built an offline evaluation harness with Recall@K and MRR.

## Engineering highlights

- Deterministic local indexing and retrieval.
- Stable document/chunk identifiers.
- Query-hash telemetry with raw-query suppression.
- Offline labeled evaluation.
- Unit, integration, CLI, privacy, and no-network tests.
- Reproducible package manifests and documented claim boundaries.

## Interview angle

This project demonstrates that I can build not only a RAG prototype, but also the measurement and observability system needed to reason about whether retrieval is actually working.
