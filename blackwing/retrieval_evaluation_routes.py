"""Per-case retrieval recall against approved, human-labeled relevant sources.

Recall@k = (labeled relevant sources found in the top k unique retrieved sources)
           / (all labeled relevant sources for that question).
The denominator comes from approved qrels, never from what a run happened to
retrieve; scoring a run against its own retrieved evidence is circular and
always reads ~100%.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from blackwing.evaluation.retrieval import query_with_case_context
from blackwing.rag.retriever import retriever

DEFAULT_QRELS_PATH = (Path(__file__).resolve().parent.parent / "evaluation_results"
                      / "recall5q_context_20261126" / "qrels_document.json")


def _read_qrels(path: Path, statuses: set[str]) -> dict[str, Any] | None:
    try:
        qrels = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    approval = qrels.get("approval")
    status = approval.get("status") if isinstance(approval, dict) else approval
    return qrels if str(status or "").upper() in statuses else None


def _load_qrels() -> dict[str, Any] | None:
    return _read_qrels(Path(os.getenv("BLACKWING_RETRIEVAL_QRELS") or DEFAULT_QRELS_PATH), {"APPROVED"})


def _case_questions(case: dict[str, Any], qrels: dict[str, Any]) -> list[dict[str, Any]]:
    incident = str(case.get("source_incident_id") or "").strip().casefold()
    if not incident:
        return []
    return [q for q in qrels.get("questions", [])
            if q.get("complete_relevance") is True and q.get("relevant_ids")
            and str((q.get("case_context") or {}).get("incident_id") or "").strip().casefold() == incident]


def _ranked_sources(question: dict[str, Any], settings) -> tuple[list[str], str]:
    """Unique source documents ranked by passage hits for one labeled question."""
    method = os.getenv("BLACKWING_AGENT_RAG_METHOD", "hybrid").strip().lower()
    if method not in {"bm25", "dense", "hybrid", "rerank"}:
        method = "hybrid"
    scope = str(question.get("scope") or "all")
    context = question.get("case_context") or {}
    filters = {"incident_date": context.get("incident_date")} if scope != "all" else {}
    result = retriever.search(
        query_with_case_context(question), [], embedding_model=settings.rag_embedding_model,
        limit=100, method=method, scope=scope, case_id=None, **filters,
    )
    ranked: list[str] = []
    seen: set[str] = set()
    for hit in result.get("results", []):
        key = str(hit.get("source_id") or hit.get("document_id") or "").strip()
        if key and key not in seen:
            seen.add(key)
            ranked.append(key)
    return ranked, str(result.get("retrieval_method") or method)


# ---------------------------------------------------------------------------
# Generic fallback for cases with no approved labels.
#
# A source counts as relevant to a case, for a retrieval scope, when BOTH hold:
#   1. Applicability: it passes the same date / jurisdiction / case-scope rules
#      the application itself applies to that scope (nothing is expected to
#      retrieve a document dated after the incident).
#   2. Topicality: it is substantially about the incident category, i.e. at least
#      TOPIC_MIN_DENSITY of its passages contain the category's key term.
# Neither condition looks at any ranking, so the denominator is independent of
# retrieval and of what a run happened to fetch. It is a rule-based proxy, not
# human judgment; approved labels, when they exist, always take precedence.
# ---------------------------------------------------------------------------
TOPIC_MIN_DENSITY = 0.05
CATEGORY_TERMS = {
    "runway_incursion": re.compile(r"incursion", re.I),
}


def _source_profiles(settings, pattern: re.Pattern) -> dict[str, dict[str, Any]]:
    """Per-source passage count, passages mentioning the topic, and one metadata row."""
    index = retriever._open_index([], settings.rag_embedding_model)
    profiles: dict[str, dict[str, Any]] = {}
    for chunk in index.chunks:
        key = str(chunk.get("source_id") or chunk.get("document_id") or "").strip()
        if not key:
            continue
        entry = profiles.setdefault(key, {"chunks": 0, "hits": 0, "row": chunk})
        entry["chunks"] += 1
        text = str(chunk.get("excerpt") or chunk.get("content") or chunk.get("text") or "")
        if pattern.search(text):
            entry["hits"] += 1
    return profiles


def _scope_kwargs(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    incident_id = str(case.get("source_incident_id") or "").strip() or None
    jurisdiction = case.get("jurisdiction") or None
    incident_date = str(case.get("incident_date") or "") or None
    kwargs = {
        "direct": {"case_id": incident_id, "jurisdiction": jurisdiction, "incident_date": incident_date},
        "historical": {"case_id": incident_id, "incident_date": incident_date},
        "guidance": {"incident_date": incident_date},
    }
    if jurisdiction and incident_date:
        kwargs["regulatory"] = {"jurisdiction": jurisdiction, "incident_date": incident_date}
    return kwargs


def _rule_based_recall(case: dict[str, Any], settings, base: dict[str, Any]) -> dict[str, Any]:
    from blackwing.rag.agent_router import TOPIC_QUERIES, _safe_query
    from blackwing.rag.engine import _scope_filter

    category = str(case.get("category") or "runway_incursion").strip().casefold()
    pattern = CATEGORY_TERMS.get(category)
    if pattern is None:
        return {**base, "status": "not_measurable",
                "reason": f"No topic definition exists for incident category '{category}'."}
    if not case.get("incident_date"):
        return {**base, "status": "not_measurable",
                "reason": "The case has no incident date, so the sources that predate it cannot be determined."}
    scope_kwargs = _scope_kwargs(case)
    method = os.getenv("BLACKWING_AGENT_RAG_METHOD", "hybrid").strip().lower()
    if method not in {"bm25", "dense", "hybrid", "rerank"}:
        method = "hybrid"
    incident_id = str(case.get("source_incident_id") or "").strip() or None

    # Rank each scope exactly as the application retrieves it.
    rankings: dict[str, list[str]] = {}
    actual_method = None
    try:
        for scope, kwargs in scope_kwargs.items():
            query = _safe_query(case.get("narrative") or "", TOPIC_QUERIES[scope],
                                case.get("evidence") or [], incident_id=incident_id or case.get("case_id"))
            result = retriever.search(query, [], embedding_model=settings.rag_embedding_model,
                                      limit=100, method=method, scope=scope, **kwargs)
            reported = str(result.get("retrieval_method") or method)
            if reported != "no_matches":
                actual_method = actual_method or reported
            ranked, seen = [], set()
            for hit in result.get("results", []):
                key = str(hit.get("source_id") or hit.get("document_id") or "").strip()
                if key and key not in seen:
                    seen.add(key)
                    ranked.append(key)
            rankings[scope] = ranked
        profiles = _source_profiles(settings, pattern)
    except Exception as exc:
        return {**base, "status": "unavailable",
                "reason": f"Could not rank the configured RAG index ({type(exc).__name__})."}

    rows = []
    for scope, kwargs in scope_kwargs.items():
        filters = {"case_id": None, "jurisdiction": None, "incident_date": None, **kwargs}
        relevant = set()
        for key, entry in profiles.items():
            row = entry["row"]
            if str(row.get("source_type") or "").casefold() == "user_submitted":
                continue
            if entry["chunks"] and entry["hits"] / entry["chunks"] >= TOPIC_MIN_DENSITY and _scope_filter(
                    row, scope=scope, include_retrospective_guidance=False, **filters):
                relevant.add(key)
        if not relevant:
            continue  # nothing in the corpus is applicable here; the scope cannot be scored
        entry = {"query_id": scope, "scope": scope, "relevant_source_count": len(relevant)}
        for k in (5, 10):
            found = len(relevant & set(rankings[scope][:k]))
            entry[f"relevant_found@{k}"] = found
            entry[f"recall@{k}"] = found / len(relevant)
        rows.append(entry)
    if not rows:
        return {**base, "status": "not_measurable", "reason": (
            "No indexed source is both applicable to this case (dated before the incident, matching "
            "jurisdiction) and about its incident category, so there is nothing to measure recall against.")}
    return {
        **base,
        "status": "completed",
        "metric": "rule_based_source_recall",
        "label_basis": "rule_based_applicability_and_topicality",
        "method": actual_method or method,
        "unit": "source documents",
        "query_count": len(rows),
        "relevant_source_count": sum(r["relevant_source_count"] for r in rows),
        "temporally_ineligible_sources": [],
        "recall@5": mean(r["recall@5"] for r in rows),
        "recall@10": mean(r["recall@10"] for r in rows),
        "queries": rows,
        "notice": (
            "No approved labels exist for this case, so relevance is defined by rule, independent of any "
            "retrieval ranking or run: a source is relevant to a scope (direct, historical, regulatory, "
            "guidance) when it passes the application's own date/jurisdiction rules for that scope and at "
            f"least {TOPIC_MIN_DENSITY:.0%} of its passages mention the incident category. Recall@k = relevant "
            "sources in the top k unique retrieved sources of that scope / all relevant sources in that scope, "
            "averaged over scopes that have any. A scope with more than k relevant sources cannot reach 100% "
            "at k. This is a rule-based proxy, not human relevance judgment."
        ),
    }


def _run_source_recall(
    case: dict[str, Any], runs: list[dict[str, Any]], settings, run_id: str | None = None,
    documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    completed = [run for run in runs if run.get("status") == "completed" and run.get("result")]
    if run_id:
        completed = [run for run in completed if run.get("run_id") == run_id]
    if not completed:
        return {"status": "not_measurable", "reason": "A completed investigation run is required."}
    latest = max(completed, key=lambda run: str(run.get("created_at") or ""))
    base = {"run_id": latest.get("run_id")}

    qrels = _load_qrels()
    questions = _case_questions(case, qrels) if qrels else []
    if not questions:
        return _rule_based_recall(case, settings, base)
    try:
        ranked_by_question = {q["question_id"]: _ranked_sources(q, settings) for q in questions}
    except Exception as exc:
        return {**base, "status": "unavailable",
                "reason": f"Could not rank the configured RAG index ({type(exc).__name__})."}

    rows = []
    actual_method = None
    for q in questions:
        ranked, method = ranked_by_question[q["question_id"]]
        actual_method = actual_method or (method if method != "no_matches" else None)
        relevant = set(q["relevant_ids"])
        row = {"query_id": q["question_id"], "query": q.get("query"),
               "relevant_source_count": len(relevant)}
        for k in (5, 10):
            found = len(relevant & set(ranked[:k]))
            row[f"relevant_found@{k}"] = found
            row[f"recall@{k}"] = found / len(relevant)
        rows.append(row)
    return {
        **base,
        "status": "completed",
        "metric": "labeled_source_recall",
        "label_basis": "approved_human_qrels",
        "scope_id": qrels.get("scope_id"),
        "method": actual_method or "hybrid",
        "unit": "source documents",
        "query_count": len(rows),
        "relevant_source_count": len({i for q in questions for i in q["relevant_ids"]}),
        "temporally_ineligible_sources": [],
        "recall@5": mean(r["recall@5"] for r in rows),
        "recall@10": mean(r["recall@10"] for r in rows),
        "queries": rows,
        "notice": (
            "Recall@k = labeled relevant sources found in the top k unique retrieved sources / all labeled "
            "relevant sources for that question, averaged over this case's labeled questions. Labels are "
            "relevance judgments made independently of what the run retrieved. Only a judged pool of sources "
            "is labeled; unlabeled sources may also be relevant, so this is not full-corpus recall."
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
        try:
            documents = repo.list_documents()
        except Exception:
            documents = []
        return {"case_id": case_id, **_run_source_recall(case, case_runs, settings, run_id, documents)}

    return router
