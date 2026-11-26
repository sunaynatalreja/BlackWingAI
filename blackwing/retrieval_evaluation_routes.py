"""Read-only API for the approved, scoped retrieval recall benchmark."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends

from blackwing.evaluation.retrieval import score_rankings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QRELS_PATH = PROJECT_ROOT / "evaluation_results/recall5q_20261123/qrels_document.json"
RANKINGS_PATH = PROJECT_ROOT / "evaluation_results/BLACKWING_EVALUATION_20261124/recall_rankings.json"


def build_router(get_settings):
    router = APIRouter(prefix="/api/v1/evaluations", tags=["evaluation"])

    @router.get(
        "/retrieval",
        summary="Get approved retrieval Recall@5 and Recall@10",
        description=(
            "Returns the read-only approved document-level BM25 benchmark when its qrels, "
            "rankings, and configured RAG index signatures match. The benchmark covers five "
            "queries and 50 labeled sources, not the full 777-source corpus or an individual run."
        ),
    )
    def retrieval_recall(settings=Depends(get_settings)):
        if not QRELS_PATH.is_file() or not RANKINGS_PATH.is_file():
            return {"status": "not_configured", "reason": "Approved retrieval qrels or rankings are unavailable."}
        index_dir = Path(settings.rag_index_dir)
        manifest_path = index_dir / "manifest.json"
        if not manifest_path.is_file():
            return {"status": "not_configured", "reason": "The configured RAG index manifest is unavailable."}

        try:
            qrels = json.loads(QRELS_PATH.read_text(encoding="utf-8"))
            rankings = json.loads(RANKINGS_PATH.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("signature") != qrels.get("index_signature"):
                return {
                    "status": "stale",
                    "reason": "The approved benchmark was labeled against a different index version; rerun retrieval evaluation for this index.",
                    "benchmark_index_signature": qrels.get("index_signature"),
                    "configured_index_signature": manifest.get("signature"),
                }
            metrics = score_rankings(qrels, rankings)
            return {"status": "completed", **metrics}
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return {"status": "unavailable", "reason": "The approved retrieval benchmark could not be validated."}

    return router
