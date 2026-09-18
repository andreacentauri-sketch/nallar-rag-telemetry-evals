# Demo

1. Run a local retrieval query with `rag_engine.py`.
2. Inspect the ranked context pack.
3. Run `evaluate.py` to reproduce Recall@1, Recall@3, and MRR.
4. Inspect a temporary JSONL telemetry file and verify it contains query hashes and result identifiers rather than raw query text.
5. Run the 12-test suite.

Suggested interview question: why use deterministic lexical retrieval here? Answer: it makes the portfolio artifact dependency-light, offline, reproducible, and easy to evaluate before adding embeddings in a later iteration.
