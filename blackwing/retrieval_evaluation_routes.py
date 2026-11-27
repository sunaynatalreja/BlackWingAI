"""Read-only API for the approved, scoped retrieval recall benchmark."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends

from blackwing.evaluation.retrieval import bm25_replay, score_rankings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QRELS_PATH = PROJECT_ROOT / "evaluation_results/recall5q_context_20261126/qrels_document.json"
RANKINGS_PATH = PROJECT_ROOT / "evaluation_results/recall5q_context_20261126/bm25_rankings.json"
QUERY_ONLY_RANKINGS_PATH = PROJECT_ROOT / "evaluation_results/recall5q_context_20261126/bm25_query_only_rankings.json"
SOURCE_HINT_BENCHMARK_PATH = PROJECT_ROOT / "evaluation_results/rag_benchmark/rag_benchmark_results.json"


def _source_hint_benchmark():
    """Summarize the saved 25-query trial as Hit@K, never mislabel it Recall."""
    if not SOURCE_HINT_BENCHMARK_PATH.is_file():
        return None
    try:
        data = json.loads(SOURCE_HINT_BENCHMARK_PATH.read_text(encoding="utf-8"))
        rows = [row for row in data.get("summary", []) if row.get("method") == "bm25"]
        if not rows:
            return None
        count = sum(int(row.get("questions_scored") or 0) for row in rows)
        if not count:
            return None
        hits5 = sum(round(float(row.get("source_hint_hit_at_5") or 0) * int(row["questions_scored"]))
                    for row in rows)
        hits10 = sum(round(float(row.get("source_hint_hit_at_10") or 0) * int(row["questions_scored"]))
                     for row in rows)
        return {
            "status": "technical_trial",
            "metric": "source-hint Hit@K, not Recall@K",
            "method": "bm25",
            "queries": count,
            "hit@5_count": hits5,
            "hit@5": hits5 / count,
            "hit@10_count": hits10,
            "hit@10": hits10 / count,
            "cases": sorted({str(row.get("case", "")).upper() for row in rows}),
            "index_chunks": data.get("indexed_chunk_count"),
            "timestamp_utc": data.get("timestamp_utc"),
            "approval_status": "SOURCE_FILTERED_TECHNICAL_TRIAL_NOT_CLEAN_HELDOUT",
        }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def build_router(get_settings):
    router = APIRouter(prefix="/api/v1/evaluations", tags=["evaluation"])

    @router.get(
        "/retrieval",
        summary="Get approved retrieval Recall@5 and Recall@10",
        description=(
            "Returns the read-only approved BM25 source-coverage benchmark when its qrels, "
            "rankings, and configured RAG index signatures match. It ranks the full non-protected "
            "source universe, while relevance labels cover only 50 reviewed sources and five queries. "
            "True full-corpus recall is unavailable until all indexed sources are judged."
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
            query_only_summary = None
            if QUERY_ONLY_RANKINGS_PATH.is_file():
                try:
                    query_only = json.loads(QUERY_ONLY_RANKINGS_PATH.read_text(encoding="utf-8"))
                    query_only_metrics = score_rankings(qrels, query_only)
                    query_only_summary = next((row for row in query_only_metrics.get("summary", [])
                                               if row.get("method") == "bm25"), None)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    query_only_summary = None
            return {
                "status": "completed", **metrics,
                "query_context_policy": qrels.get("query_context_policy"),
                "query_only_control": query_only_summary,
                "source_hint_trial": _source_hint_benchmark(),
            }
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return {"status": "unavailable", "reason": "The approved retrieval benchmark could not be validated."}

    return router
