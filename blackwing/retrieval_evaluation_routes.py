"""Per-case retrieval coverage against the evidence supplied to its agents."""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from blackwing.rag.query_text import case_query_text
from blackwing.rag.retriever import retriever


def _workflow_evidence(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Recover the persisted, pre-clipping evidence packet for a completed run."""
    snapshot = run.get("evaluation_input_snapshot") or {}
    rows = snapshot.get("evidence") or []
    if rows:
        return [row for row in rows if isinstance(row, dict)]

    # Compatibility for runs created before evaluation_input_snapshot existed.
    resume = run.get("resume_payload") or {}
    rows = list(resume.get("retrieved") or [])
    packets = resume.get("agent_evidence") or {}
    if isinstance(packets, dict):
        for packet in packets.values():
            rows.extend(packet if isinstance(packet, list) else [])
    return [row for row in rows if isinstance(row, dict)]


def _indexed_sources(rows: list[dict[str, Any]]) -> set[str]:
    """Use only non-user corpus sources that were present in the agent packet."""
    sources: set[str] = set()
    for row in rows:
        source_id = str(row.get("source_id") or "").strip()
        source_type = str(row.get("source_type") or "").strip().casefold()
        role = str(row.get("evidence_role") or "").strip().casefold()
        if not source_id or source_id.casefold().startswith("case-narrative:"):
            continue
        if source_type in {"user_submitted", "user_report", "incident_narrative"} or role == "reported":
            continue
        if source_id.casefold().startswith("user supplied source:"):
            continue
        sources.add(source_id)
    return sources


def _ranked_sources(case: dict[str, Any], settings) -> tuple[list[str], str]:
    """Rank existing corpus sources using incident narrative, never the case ID."""
    method = os.getenv("BLACKWING_AGENT_RAG_METHOD", "hybrid").strip().lower()
    if method not in {"bm25", "dense", "hybrid", "rerank"}:
        method = "hybrid"
    narrative_query = case_query_text(str(case.get("narrative") or ""))
    query = " ".join(part for part in (
        narrative_query,
        str(case.get("location") or "").strip(),
        "runway incursion",
    ) if part)
    result = retriever.search(
        query,
        [],
        embedding_model=settings.rag_embedding_model,
        limit=100,
        method=method,
        scope="all",
        case_id=None,
        jurisdiction=case.get("jurisdiction"),
        incident_date=str(case.get("incident_date") or "") or None,
    )
    ranked: list[str] = []
    seen: set[str] = set()
    for hit in result.get("results", []):
        source_id = str(hit.get("source_id") or hit.get("document_id") or "").strip()
        if source_id and source_id not in seen:
            seen.add(source_id)
            ranked.append(source_id)
    return ranked, str(result.get("retrieval_method") or method)


def _run_source_recall(
    case: dict[str, Any], runs: list[dict[str, Any]], settings, run_id: str | None = None,
) -> dict[str, Any]:
    completed = [run for run in runs if run.get("status") == "completed" and run.get("result")]
    if run_id:
        completed = [run for run in completed if run.get("run_id") == run_id]
    if not completed:
        return {"status": "not_measurable", "reason": "A completed investigation run is required to evaluate its agent evidence."}
    latest = max(completed, key=lambda run: str(run.get("created_at") or ""))
    relevant = _indexed_sources(_workflow_evidence(latest))
    if not relevant:
        return {
            "status": "not_measurable",
            "reason": "The completed run has no persisted non-user corpus sources in its agent evidence packet.",
            "run_id": latest.get("run_id"),
        }
    try:
        ranked, actual_method = _ranked_sources(case, settings)
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"Could not rank the configured RAG index ({type(exc).__name__}).",
            "run_id": latest.get("run_id"),
        }

    rows = []
    for k in (5, 10):
        found = len(relevant.intersection(ranked[:k]))
        rows.append({
            "query_id": "incident_narrative",
            "relevant_source_count": len(relevant),
            f"relevant_found@{k}": found,
            f"recall@{k}": found / len(relevant),
        })
    by_k = {5: rows[0], 10: rows[1]}
    return {
        "status": "completed",
        "run_id": latest.get("run_id"),
        "metric": "agent_evidence_source_recall_proxy",
        "label_basis": "unique_non_user_sources_in_completed_run_evidence_packet",
        "method": actual_method,
        "actual_methods": [actual_method],
        "unit": "unique indexed source documents",
        "query_count": 1,
        "relevant_source_count": len(relevant),
        "recall@5": by_k[5]["recall@5"],
        "recall@10": by_k[10]["recall@10"],
        "queries": [
            {
                "query_id": "incident_narrative",
                "relevant_source_count": len(relevant),
                "relevant_found@5": by_k[5]["relevant_found@5"],
                "recall@5": by_k[5]["recall@5"],
                "relevant_found@10": by_k[10]["relevant_found@10"],
                "recall@10": by_k[10]["recall@10"],
            }
        ],
        "notice": (
            "Post-hoc agent-evidence recall proxy: the relevant set is the unique indexed corpus "
            "sources present in the completed run's pre-clipping agent evidence packet; uploaded "
            "case evidence and incident IDs are excluded. The narrative is searched against the "
            "current index and its top 5/10 unique sources are compared with that set. This is not "
            "independent human qrels or full-corpus recall, and can be optimistic because the set "
            "was assembled by the investigation workflow."
        ),
    }


def build_router(get_settings, get_repository):
    router = APIRouter(prefix="/api/v1/cases", tags=["evaluation"])

    @router.get(
        "/{case_id}/retrieval-evaluation",
        summary="Evaluate narrative retrieval coverage at 5 and 10",
        description=(
            "Ranks existing index sources using the case narrative (not its incident ID), then "
            "compares top-5/top-10 sources with unique non-user corpus sources persisted in the "
            "latest completed run's agent evidence packet. Returns an explicitly labeled "
            "post-hoc agent-evidence recall proxy, not human-qrels or full-corpus recall."
        ),
        responses={404: {"description": "Case not found"}},
    )
    def case_retrieval_evaluation(
        case_id: str, run_id: str | None = None,
        settings=Depends(get_settings), repo=Depends(get_repository),
    ):
        case = repo.get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="Case not found")
        case_runs = repo.list_case_runs(case_id)
        if run_id and not any(run.get("run_id") == run_id for run in case_runs):
            raise HTTPException(status_code=404, detail="Run not found for this case")
        return {"case_id": case_id, **_run_source_recall(case, case_runs, settings, run_id)}

    return router
