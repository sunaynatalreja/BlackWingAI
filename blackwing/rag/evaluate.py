from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from .embeddings import make_embeddings
from .engine import CrossEncoderReranker, HybridIndex


def _is_true(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def metrics_for_ranking(ranked: list[str], relevant: set[str], *, complete: bool, k: int) -> dict[str, float | None]:
    if k < 1:
        raise ValueError("k must be positive")
    unique = list(dict.fromkeys(ranked))[:k]
    found = len(set(unique) & relevant)
    return {"recall": found / len(relevant) if complete and relevant else None,
            "precision": found / k if complete else None,
            "hit": 1.0 if found else 0.0,
            "mrr": next((1 / rank for rank, row in enumerate(unique, 1) if row in relevant), 0.0)}


def validate_citation_ids(cited: list[str], indexed_chunk_ids: set[str]) -> dict[str, Any]:
    """Structural citation ID verification only. NOT semantic claim support."""
    return {"valid_count": sum(cid in indexed_chunk_ids for cid in cited),
            "total_count": len(cited), "invalid_citations": [cid for cid in cited if cid not in indexed_chunk_ids],
            "semantic_citation_precision": "NOT VERIFIED: needs claim-level human judgments"}


def evaluate(index: HybridIndex, gold_rows: list[dict[str, Any]], *,
             methods: list[str], reranker: Any = None, k_values: tuple[int, ...] = (5, 10)) -> dict[str, Any]:
    """Do not label Hit@k or a page-location proxy as true Recall@k."""
    output: dict[str, Any] = {"methods": {}, "queries": [],
        "end_to_end_answer_accuracy": "NOT VERIFIED: separately judge Qwen outputs",
        "citation_precision": "NOT VERIFIED: separately judge claim-level support"}
    max_k = max(k_values)
    for method in methods:
        scores = {f"recall@{k}": [] for k in k_values}
        scores.update({f"precision@{k}": [] for k in k_values})
        scores.update({f"hit@{k}": [] for k in k_values})
        scores["mrr@10"] = []
        for item in gold_rows:
            if not item.get("question_id") or not item.get("query"):
                raise ValueError("Gold rows must have question_id and query")
            relevant = set(str(item.get("relevant_chunk_ids") or "").split("|")) - {""}
            complete = _is_true(item.get("complete_relevance"))
            # Complete relevance labels must be substantive, not an empty declaration.
            if complete and not relevant:
                raise ValueError("Complete judgments require nonempty relevant_chunk_ids")
            result = index.search(item["query"], top_k=max_k, method=method, reranker=reranker,
                    scope=item.get("scope") or "all", case_id=item.get("case_id") or None,
                    jurisdiction=item.get("jurisdiction") or None,
                    incident_date=item.get("incident_date") or None)
            ids = [row["chunk_id"] for row in result["results"]]
            record = {"question_id": item["question_id"], "method": method,
                      "split": item.get("split", "unassigned"), "ranked_ids": ids,
                      "complete_relevance": complete,
                      "retrieval_method": result["retrieval_method"]}
            for k in k_values:
                m = metrics_for_ranking(ids, relevant, complete=complete, k=k)
                for metric in ("hit", "precision", "recall"):
                    label = f"{metric}@{k}"
                    record[label] = m[metric]
                    if m[metric] is not None:
                        scores[label].append(m[metric])
            record["mrr@10"] = metrics_for_ranking(ids, relevant, complete=complete, k=10)["mrr"]
            if relevant:
                scores["mrr@10"].append(record["mrr@10"])
            output["queries"].append(record)
        output["methods"][method] = {
            metric: round(mean(values), 4) if values else None for metric, values in scores.items()
        } | {"questions_total": len(gold_rows),
             "completely_judged": sum(_is_true(x.get("complete_relevance")) for x in gold_rows),
             "recall_status": "MEASURED" if all(_is_true(x.get("complete_relevance")) for x in gold_rows) and gold_rows
                               else "PARTIAL_OR_NOT_VERIFIED",
             "method_verified": method != "rerank" or reranker is not None}
    return output


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New output JSON; refuses overwrite")
    p.add_argument("--backend", choices=["none", "local", "iit"], default="iit")
    p.add_argument("--rerank", action="store_true", help="Use actual CrossEncoder (may download weights)")
    p.add_argument("--cache", type=Path)
    args = p.parse_args(argv)
    if args.output.exists():
        p.error("Output exists: previous evaluation will not be overwritten")
    embeddings = make_embeddings(args.backend, cache_path=args.cache)
    index = HybridIndex.load(args.index, embeddings=embeddings)
    with args.gold.open(encoding="utf-8-sig", newline="") as f:
        gold = list(csv.DictReader(f))
    methods = ["bm25"] if embeddings is None or index.vectors is None else ["bm25", "dense", "hybrid"]
    reranker = CrossEncoderReranker() if args.rerank else None
    if reranker:
        methods.append("rerank")
    result = evaluate(index, gold, methods=methods, reranker=reranker)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": result["methods"], "output": str(args.output)}, indent=2))
    if not args.rerank:
        print("CrossEncoder comparison NOT VERIFIED: rerank option was not selected")


if __name__ == "__main__":
    main()
