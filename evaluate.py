#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from rag_engine import RAGService

def load_eval(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError("eval set must be list")
    return obj

def evaluate(corpus: Path, eval_path: Path, telemetry: Path, k: int = 3):
    svc = RAGService(corpus, telemetry)
    cases = load_eval(eval_path)
    hit1 = 0
    hitk = 0
    rr_sum = 0.0
    rows = []
    for row in cases:
        qid = row["id"]
        query = row["query"]
        expected = row["expected_doc_id"]
        results = svc.retrieve(query, k=k)
        doc_ids = [r["doc_id"] for r in results]
        rank = None
        for i, doc_id in enumerate(doc_ids, 1):
            if doc_id == expected:
                rank = i
                break
        hit1 += int(rank == 1)
        hitk += int(rank is not None)
        rr_sum += (1.0 / rank) if rank else 0.0
        rows.append({
            "id": qid,
            "expected_doc_id": expected,
            "top_doc_id": doc_ids[0] if doc_ids else None,
            "rank_of_expected": rank,
            "hit_at_1": rank == 1,
            "hit_at_k": rank is not None,
        })
    n = len(cases)
    return {
        "case_count": n,
        "k": k,
        "recall_at_1": hit1 / n,
        "recall_at_k": hitk / n,
        "mrr": rr_sum / n,
        "cases": rows,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--telemetry", required=True)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    result = evaluate(Path(args.corpus), Path(args.eval), Path(args.telemetry), args.k)
    print(json.dumps(result, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
