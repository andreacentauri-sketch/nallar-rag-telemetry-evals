# Evaluation

The bundled offline evaluation set contains eight labeled retrieval questions covering the portfolio corpus.

Primary metrics:

- Recall@1
- Recall@K (K=3)
- Mean Reciprocal Rank (MRR)

Giantstep thresholds:

- Recall@3 >= 0.875
- MRR >= 0.80
- 12/12 unit/integration tests PASS
- telemetry privacy test PASS

These metrics establish retrieval behavior on the bundled offline evaluation set. They do not establish production-domain accuracy, generative answer correctness, or production promotion.
