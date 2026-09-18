#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

def tokenize(text: str) -> List[str]:
    return [x.lower() for x in TOKEN_RE.findall(text)]

def chunk_text(text: str, chunk_tokens: int = 80, overlap_tokens: int = 20) -> List[str]:
    toks = tokenize(text)
    if not toks:
        return []
    if chunk_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("invalid chunk parameters")
    out = []
    step = chunk_tokens - overlap_tokens
    for start in range(0, len(toks), step):
        piece = toks[start:start + chunk_tokens]
        if not piece:
            break
        out.append(" ".join(piece))
        if start + chunk_tokens >= len(toks):
            break
    return out

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    token_count: int

class TelemetrySink:
    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None

    def emit(self, event: Dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")

class BM25Index:
    def __init__(self, chunks: List[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("index requires chunks")
        self.chunks = chunks
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_tokens = [tokenize(c.text) for c in chunks]
        self.tf = [Counter(toks) for toks in self.doc_tokens]
        self.lengths = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.lengths) / len(self.lengths)
        df = Counter()
        for toks in self.doc_tokens:
            for term in set(toks):
                df[term] += 1
        n = len(chunks)
        self.idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query: str, idx: int) -> float:
        q = tokenize(query)
        tf = self.tf[idx]
        dl = self.lengths[idx]
        score = 0.0
        for term in q:
            if term not in tf:
                continue
            freq = tf[term]
            idf = self.idf.get(term, 0.0)
            denom = freq + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            score += idf * (freq * (self.k1 + 1.0)) / denom
        return score

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if k <= 0:
            raise ValueError("k must be positive")
        ranked = []
        for i, chunk in enumerate(self.chunks):
            score = self.score(query, i)
            ranked.append((score, chunk.chunk_id, i))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        out = []
        for rank, (score, _cid, i) in enumerate(ranked[:k], 1):
            c = self.chunks[i]
            out.append({
                "rank": rank,
                "score": round(score, 8),
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "title": c.title,
                "text": c.text,
            })
        return out

def load_corpus(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError("corpus must be a list")
    out = []
    seen = set()
    for row in obj:
        if not isinstance(row, dict):
            raise ValueError("corpus row must be object")
        doc_id = row.get("doc_id")
        title = row.get("title")
        text = row.get("text")
        if not all(isinstance(x, str) and x.strip() for x in (doc_id, title, text)):
            raise ValueError("invalid corpus row")
        if doc_id in seen:
            raise ValueError("duplicate doc_id")
        seen.add(doc_id)
        out.append({"doc_id": doc_id, "title": title, "text": text})
    return out

def build_chunks(corpus: List[Dict[str, Any]], chunk_tokens: int = 80, overlap_tokens: int = 20) -> List[Chunk]:
    chunks = []
    for doc in corpus:
        pieces = chunk_text(doc["text"], chunk_tokens, overlap_tokens)
        for i, piece in enumerate(pieces):
            chunks.append(Chunk(
                chunk_id=f'{doc["doc_id"]}::c{i:03d}',
                doc_id=doc["doc_id"],
                title=doc["title"],
                text=piece,
                token_count=len(tokenize(piece)),
            ))
    return chunks

class RAGService:
    def __init__(
        self,
        corpus_path: Path,
        telemetry_path: Optional[Path] = None,
        chunk_tokens: int = 80,
        overlap_tokens: int = 20,
    ):
        started = time.perf_counter()
        self.corpus_path = Path(corpus_path)
        self.telemetry = TelemetrySink(telemetry_path)
        self.corpus = load_corpus(self.corpus_path)
        self.chunks = build_chunks(self.corpus, chunk_tokens, overlap_tokens)
        self.index = BM25Index(self.chunks)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.telemetry.emit({
            "event": "index_build",
            "doc_count": len(self.corpus),
            "chunk_count": len(self.chunks),
            "elapsed_ms": round(elapsed_ms, 3),
        })

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        started = time.perf_counter()
        results = self.index.search(query, k=k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.telemetry.emit({
            "event": "retrieval",
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "query_length": len(query),
            "k": k,
            "elapsed_ms": round(elapsed_ms, 3),
            "result_doc_ids": [r["doc_id"] for r in results],
            "result_chunk_ids": [r["chunk_id"] for r in results],
            "raw_query_persisted": False,
        })
        return results

    def context_pack(self, query: str, k: int = 3) -> Dict[str, Any]:
        results = self.retrieve(query, k=k)
        return {
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "contexts": [{
                "rank": r["rank"],
                "doc_id": r["doc_id"],
                "chunk_id": r["chunk_id"],
                "title": r["title"],
                "text": r["text"],
            } for r in results],
        }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--telemetry", default=None)
    args = ap.parse_args()
    svc = RAGService(Path(args.corpus), Path(args.telemetry) if args.telemetry else None)
    pack = svc.context_pack(args.query, args.k)
    print(json.dumps(pack, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
