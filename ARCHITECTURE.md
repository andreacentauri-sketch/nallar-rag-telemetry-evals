# Architecture

```text
JSON corpus
   |
   v
tokenize + chunk
   |
   v
deterministic BM25 index
   |
query ---> retrieve top-k ---> context pack
   |             |
   |             `----> ranked doc/chunk ids
   v
privacy-preserving telemetry JSONL
(query hash, latency, k, ids; no raw query)
   |
   v
offline evaluation harness
Recall@1 / Recall@K / MRR
```

The project is self-contained and does not mutate canonical NALLAR runtime source.
