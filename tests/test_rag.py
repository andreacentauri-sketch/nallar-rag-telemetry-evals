from __future__ import annotations
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag_engine import tokenize, chunk_text, load_corpus, build_chunks, BM25Index, RAGService
from evaluate import evaluate

CORPUS = ROOT / "data" / "sample_corpus.json"
EVAL = ROOT / "data" / "eval_set.json"

class TestRAG(unittest.TestCase):
    def test_01_tokenize(self):
        self.assertEqual(tokenize("FastAPI + JSON-RPC 2.0"), ["fastapi","json","rpc","2","0"])

    def test_02_chunk_overlap_validation(self):
        with self.assertRaises(ValueError):
            chunk_text("a b c", 3, 3)

    def test_03_corpus_load(self):
        corpus = load_corpus(CORPUS)
        self.assertGreaterEqual(len(corpus), 8)
        self.assertEqual(len({d["doc_id"] for d in corpus}), len(corpus))

    def test_04_chunks_build(self):
        chunks = build_chunks(load_corpus(CORPUS), 40, 10)
        self.assertGreaterEqual(len(chunks), len(load_corpus(CORPUS)))

    def test_05_bm25_target(self):
        chunks = build_chunks(load_corpus(CORPUS), 80, 20)
        index = BM25Index(chunks)
        top = index.search("stdio JSON RPC MCP tools resources prompts", 1)[0]
        self.assertEqual(top["doc_id"], "mcp_server")

    def test_06_service_retrieve(self):
        svc = RAGService(CORPUS)
        top = svc.retrieve("query hash telemetry raw query privacy", 1)[0]
        self.assertEqual(top["doc_id"], "telemetry")

    def test_07_context_pack(self):
        svc = RAGService(CORPUS)
        pack = svc.context_pack("typed FastAPI streaming local model Ollama", 2)
        self.assertEqual(len(pack["contexts"]), 2)
        self.assertIn("query_sha256", pack)

    def test_08_telemetry_privacy(self):
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td) / "telemetry.jsonl"
            query = "private synthetic query that must not be persisted"
            svc = RAGService(CORPUS, tp)
            svc.retrieve(query, 3)
            text = tp.read_text(encoding="utf-8")
            self.assertNotIn(query, text)
            rows = [json.loads(x) for x in text.splitlines()]
            retrieval = [x for x in rows if x["event"] == "retrieval"][0]
            self.assertFalse(retrieval["raw_query_persisted"])
            self.assertEqual(len(retrieval["query_sha256"]), 64)

    def test_09_telemetry_schema(self):
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td) / "telemetry.jsonl"
            svc = RAGService(CORPUS, tp)
            svc.retrieve("recall mrr evaluation", 3)
            rows = [json.loads(x) for x in tp.read_text().splitlines()]
            self.assertEqual(rows[0]["event"], "index_build")
            self.assertEqual(rows[1]["event"], "retrieval")
            self.assertIn("elapsed_ms", rows[1])
            self.assertIn("result_doc_ids", rows[1])

    def test_10_eval_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            result = evaluate(CORPUS, EVAL, Path(td) / "telemetry.jsonl", 3)
            self.assertEqual(result["case_count"], 8)
            self.assertGreaterEqual(result["recall_at_k"], 0.875)
            self.assertGreaterEqual(result["mrr"], 0.80)

    def test_11_cli(self):
        p = subprocess.run(
            [sys.executable, str(ROOT/"rag_engine.py"), "--corpus", str(CORPUS),
             "--query", "MCP stdio JSON RPC", "--k", "2"],
            capture_output=True, text=True, timeout=30
        )
        self.assertEqual(p.returncode, 0)
        obj = json.loads(p.stdout)
        self.assertEqual(len(obj["contexts"]), 2)

    def test_12_no_network_dependency(self):
        src = (ROOT/"rag_engine.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("import requests", src)
        self.assertNotIn("import urllib", src)
        self.assertNotIn("import socket", src)

if __name__ == "__main__":
    unittest.main(verbosity=2)
