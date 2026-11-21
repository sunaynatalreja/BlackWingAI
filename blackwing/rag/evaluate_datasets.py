"""Run unseen/golden prompts against the knowledge-base index (retrieval only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .datasets import DATASET_ROOT, load_demo_inputs
from .embeddings import make_embeddings
from .engine import HybridIndex


def evaluate_dataset(index: HybridIndex, cases: list[dict[str, Any]], *, method: str, limit: int) -> dict[str, Any]:
    output: dict[str, Any] = {
        "evaluation_type": "retrieval_only",
        "answer_quality": "NOT MEASURED: this command does not run agents or inspect protected gold answers",
        "index_signature": index.signature,
        "cases": [],
    }
    for case in cases:
        result: dict[str, Any] = {key: case[key] for key in (
            "case_id", "case_name", "dataset_split", "input_file", "validation_status"
        )}
        if not case["narrative"]:
            output["cases"].append(result)
            continue
        routes: dict[str, Any] = {}
        for scope in ("historical", "regulatory", "guidance"):
            if scope == "regulatory" and not (case["jurisdiction"] and case["incident_date"]):
                routes[scope] = {"status": "skipped_missing_jurisdiction_or_date", "sources": []}
                continue
            found = index.search(
                case["narrative"], top_k=limit, method=method, scope=scope,
                case_id=case["case_id"], jurisdiction=case["jurisdiction"],
                incident_date=case["incident_date"],
            )
            routes[scope] = {
                "retrieval_method": found["retrieval_method"],
                "limitations": found["limitations"],
                "sources": [
                    {"scope": scope, **{key: row.get(key) for key in (
                        "chunk_id", "source_id", "document_id", "title", "source_type",
                        "incident_id", "jurisdiction", "page_number", "excerpt",
                    )}}
                    for row in found["results"]
                ],
            }
        result["routes"] = routes
        output["cases"].append(result)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--split", choices=("unseen", "golden"), default="unseen")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("bm25", "dense", "hybrid", "rerank"), default="hybrid")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--backend", choices=("none", "local", "iit"), default="iit")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Refusing to overwrite an existing dataset evaluation")
    if not 1 <= args.limit <= 20:
        parser.error("--limit must be between 1 and 20")
    embedder = make_embeddings(args.backend, cache_path=args.cache)
    index = HybridIndex.load(args.index, embeddings=embedder)
    cases = load_demo_inputs(args.split, dataset_root=args.dataset_root)
    result = evaluate_dataset(index, cases, method=args.method, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "split": args.split,
        "cases": len(cases),
        "with_narrative": sum(bool(row["narrative"]) for row in cases),
        "retrieval_metrics": "No relevance labels supplied; inspect/annotate retrieval before reporting recall@k",
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
