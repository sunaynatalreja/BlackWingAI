from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .corpus import load_manifest
from .engine import HybridIndex
from .tokenization import PLANS, BGESubwordTokenizer
from .embeddings import make_embeddings
from .engine import CrossEncoderReranker


def compare(documents: list[dict[str, Any]], questions: list[dict[str, str]], tokenizer: Any,
            *, embeddings: Any = None, method: str = "bm25", reranker: Any = None) -> dict[str, Any]:
    if not questions:
        raise ValueError("Provide nonempty development questions")
    if any(str(row.get("split", "")).strip().casefold() not in {"development", "dev"} for row in questions):
        raise ValueError("Chunking tuning may use only the DEVELOPMENT split")
    if method not in {"bm25", "hybrid", "rerank"}:
        raise ValueError("Chunking method must be bm25, hybrid or rerank")
    if method != "bm25" and embeddings is None:
        raise ValueError("Hybrid/Rerank chunking tests require a real matching embedding backend")
    if method == "rerank" and reranker is None:
        raise ValueError("CrossEncoder reranker is required for the rerank chunking experiment")
    out = {}
    for strategy in PLANS:
        index = HybridIndex.from_documents(documents, strategy=strategy, tokenizer=tokenizer, embeddings=embeddings)
        ranks = []
        for item in questions:
            expected = set(item.get("expected_source_pages", "").split("|")) - {""}
            if not expected:
                raise ValueError("Expected source-page judgments required for chunking selection")
            results = index.search(item["query"], top_k=10, method=method, reranker=reranker)["results"]
            rank = next((n for n, hit in enumerate(results, 1) if
                         f"{hit.get('source_id')}#{hit.get('page_number')}" in expected), None)
            ranks.append(rank)
        out[strategy] = {
            "chunks": len(index.chunks),
            "expected_page_hit_at_5": round(mean(1.0 if r and r <= 5 else 0.0 for r in ranks), 4),
            "expected_page_hit_at_10": round(mean(1.0 if r else 0.0 for r in ranks), 4),
            "expected_page_mrr_at_10": round(mean(1 / r if r else 0 for r in ranks), 4),
            "retrieval_recall_at_k": "NOT VERIFIED: source-page hit is only a proxy",
            "tokenizer_verified": bool(getattr(tokenizer, "verified", False)),
            "question_count": len(questions),
            "method": method,
        }
    best = max(out, key=lambda k: (out[k]["expected_page_mrr_at_10"],
                                   out[k]["expected_page_hit_at_5"], -out[k]["chunks"]))
    return {"methods": out, "proposed_strategy_from_development_proxy": best,
            "warning": "Validate selected strategy on separate held-out questions; never claim source-page hit is true recall"}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--development-gold", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=("none", "local", "iit"), default="iit",
                   help="Hosted IIT embeddings by default; requires configured key and CA")
    p.add_argument("--method", choices=("bm25", "hybrid", "rerank"), default="bm25")
    p.add_argument("--cache", type=Path)
    p.add_argument("--confirm-hosted-embeddings", action="store_true")
    a = p.parse_args(argv)
    if a.output.exists():
        p.error("Refusing to overwrite prior experiment results")
    docs, audit = load_manifest(a.manifest, a.root)
    with a.development_gold.open(encoding="utf-8-sig", newline="") as fh:
        gold = list(csv.DictReader(fh))
    if a.backend == "iit" and not a.confirm_hosted_embeddings:
        p.error("IIT-hosted embeddings require --confirm-hosted-embeddings (uses team quota)")
    embedding = make_embeddings(a.backend, cache_path=a.cache)
    tokenizer = embedding.tokenizer if embedding else BGESubwordTokenizer()
    reranker = CrossEncoderReranker() if a.method == "rerank" else None
    result = compare(docs, gold, tokenizer, embeddings=embedding, method=a.method,
                     reranker=reranker)
    result["source_manifest_audit"] = {"approved_document_count": audit.approved_document_count,
                                       "asrs_iit_counting": audit.asrs_iit_counting}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
