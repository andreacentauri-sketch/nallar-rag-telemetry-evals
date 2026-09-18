# Telemetry

Two event types are emitted:

- `index_build`: document count, chunk count, elapsed milliseconds.
- `retrieval`: query SHA-256, query length, `k`, elapsed milliseconds, result document IDs, result chunk IDs, and `raw_query_persisted=false`.

The default implementation never writes raw query text to telemetry. A test explicitly verifies this boundary.
