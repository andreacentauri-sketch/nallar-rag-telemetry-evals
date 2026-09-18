# RAG + Telemetry + Evals

> Recruiter-facing public presentation generated from the verified local NALLAR portfolio package.

## 30-second summary

This project is presented around four questions: **what was built, how it was tested, what was measured, and what is not being claimed**.

## Evidence model

- Runnable implementation or portfolio artifact.
- Executable tests and/or quantitative evaluation.
- Reproducible local commands.
- Explicit limitations.

## Proof points

- Deterministic local BM25-style retrieval.
- Privacy-preserving telemetry.
- Offline labeled evaluation harness.
- **12/12 tests passed.**
- **Recall@1 = 1.00**
- **Recall@3 = 1.00**
- **MRR = 1.00**

## Recruiter demo

Run one retrieval query, then run the evaluation harness and show the metrics file.

## Evidence boundary

Metrics apply to the bundled labeled evaluation set; they are not claims of general production-domain accuracy.

---

## Reproduce locally

See the project files and original run notes below. Use the repository's own test/evaluation commands and inspect the generated evidence artifacts rather than relying on screenshots alone.


### Original run notes

# NALLAR RAG + Telemetry + Evals

A self-contained local retrieval portfolio project with deterministic BM25 ranking, structured telemetry, and an offline evaluation harness.

## Run retrieval

```bash
cd ~/Downloads/NALLAR_AI_ENGINEERING_PORTFOLIO/03_RAG_TELEMETRY_EVALS
python rag_engine.py --corpus data/sample_corpus.json --query "MCP stdio JSON RPC" --k 3
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Run evaluation

```bash
python evaluate.py --corpus data/sample_corpus.json --eval data/eval_set.json --telemetry /tmp/nallar_rag_telemetry.jsonl --k 3
```

This package uses only the Python standard library and makes no network calls.


## Public claim boundary

This repository is published as an evidence-backed portfolio artifact. Test and evaluation results apply to the documented local/bundled scope. No production deployment, foundation-model training, or other unverified capability is implied.
