from __future__ import annotations

from datetime import datetime, timezone
import json
import asyncio
import logging
import os
from typing import Any, Awaitable, Callable
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Response, status

from blackwing.config import Settings
from blackwing.repository import MemoryRepository, repository
from blackwing.schemas import (
    CaseCreate,
    CaseListResponse,
    CasePatch,
    CaseResponse,
    EvidenceInput,
    EvidenceResponse,
    KnowledgeDocumentCreate,
    KnowledgeDocumentResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    ReportResponse,
    ReviewCreate,
    ReviewResponse,
    RunCreate,
    RunResponse,
    TraceResponse,
)
from blackwing.agents.base import AgentExecutionError
from blackwing.service import analyze_incident
from blackwing.rag.retriever import retriever
from blackwing.tools import retrieve_evidence, tool_audit
from blackwing.diagnostics import lifecycle_event, safe_error

AnalysisRunner = Callable[..., Awaitable[dict[str, Any]]]
logger = logging.getLogger("blackwing.runs")

app = FastAPI(
    title="BlackWing AI API",
    description="Versioned case, evidence, analysis, review, and trace API. Draft 1 supports runway incursions.",
    version="1.0.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
)


def get_repository() -> MemoryRepository:
    return repository


def get_settings() -> Settings:
    try:
        return Settings.load()
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail=f"Application configuration unavailable: {exc}") from exc


def get_analysis_runner() -> AnalysisRunner:
    return analyze_incident


def _case_or_404(repo: MemoryRepository, case_id: str) -> dict[str, Any]:
    case = repo.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


def _run_links(run_id: str, case_id: str) -> dict[str, str]:
    return {
        "self": f"/api/v1/runs/{run_id}",
        "trace": f"/api/v1/runs/{run_id}/trace",
        "report": f"/api/v1/cases/{case_id}/report",
    }


def _run_response(run: dict[str, Any]) -> RunResponse:
    return RunResponse(
        **{key: run[key] for key in (
            "run_id", "case_id", "status", "workflow", "model", "progress",
            "error_code", "error", "created_at", "started_at", "completed_at",
        )},
        links=_run_links(run["run_id"], run["case_id"]),
    )


async def _execute_run(
    run_id: str,
    case: dict[str, Any],
    retrieved: list[dict[str, Any]],
    settings: Settings,
    repo: MemoryRepository,
    runner: AnalysisRunner,
    use_agent_retrieval: bool = True,
) -> None:
    """Contain background failures; cooperative server shutdown still propagates."""
    try:
        await _execute_run_body(run_id, case, retrieved, settings, repo, runner, use_agent_retrieval)
    except asyncio.CancelledError as exc:
        _record_run_failure(run_id, case, repo, exc, interrupted=True)
        raise
    except (Exception, SystemExit) as exc:
        _record_run_failure(run_id, case, repo, exc)


def _record_run_failure(run_id, case, repo, exc, *, interrupted=False):
    diagnostic = safe_error(exc, "interrupted" if interrupted else None)
    logger.error("Background investigation stopped; run=%s code=%s error_type=%s", run_id, diagnostic["code"], diagnostic["type"])
    event = lifecycle_event("workflow_execution", "execution_interrupted" if interrupted else "execution_failed",
                            case_id=case.get("case_id"), error=diagnostic)
    # Error handling must not become a second unhandled background exception.
    # Disk/storage failures may prevent durability; log their types explicitly.
    operations = [("append_failure_event", lambda: getattr(repo, "append_event", lambda *_: None)(run_id, event))]
    operations.extend([
        ("patch_failed_run", lambda: repo.patch_run(run_id, {"status": "failed", "error_code": diagnostic["code"],
            "error": diagnostic["message"], "completed_at": datetime.now(timezone.utc)})),
        ("update_case", lambda: repo.update_case(case["case_id"], {"status": "analysis_failed"})),
    ])
    for operation, persist in operations:
        try:
            persist()
        except Exception as storage_exc:
            logger.error("Could not persist background failure; run=%s operation=%s error_type=%s", run_id, operation, type(storage_exc).__name__)


async def _execute_run_body(
    run_id: str,
    case: dict[str, Any],
    retrieved: list[dict[str, Any]],
    settings: Settings,
    repo: MemoryRepository,
    runner: AnalysisRunner,
    use_agent_retrieval: bool = True,
) -> None:
    repo.patch_run(run_id, {"status": "running", "started_at": datetime.now(timezone.utc)})
    evidence = [*case.get("evidence", []), *retrieved]
    stored = repo.get_run(run_id) if hasattr(repo, "get_run") else {}
    invocation_events = list((stored or {}).get("preflight_trace", []))
    def record_event(event):
        invocation_events.append(event)
        if hasattr(repo, "append_event"):
            repo.append_event(run_id, event)
        else:
            repo.patch_run(run_id, {"audit_events": list(invocation_events)})
    def checkpoint(state):
        repo.patch_run(run_id, {"partial_outputs": state.get("outputs", {}), "plan": state.get("plan"),
                               "execution_steps": state.get("execution_steps", 0),
                               "progress": [{"step": e["step"], "status": e["status"]} for e in state.get("trace", [])] + [e for e in invocation_events if e.get("event_id")]})
        record_event({"step": "state_update", "status": "complete", "detail": state.get("trace", [])[-1:]})
    record_event(lifecycle_event("workflow_execution", "execution_started", case_id=case["case_id"]))
    agent_evidence = None
    if use_agent_retrieval and os.getenv("BLACKWING_AGENT_RAG", "0") == "1":
        # Keep integration outside agent packages and UI. No LLM calls here.
        from blackwing.rag.agent_router import build_agent_packets
        try:
            with tool_audit(case["case_id"], record_event):
                agent_evidence = retrieve_evidence(build_agent_packets, input_identifiers=[case["case_id"]],
                    retriever=retriever, narrative=case["narrative"],
                    case_id=case.get("source_incident_id") or case["case_id"],
                    jurisdiction=case.get("jurisdiction"), incident_date=str(case.get("incident_date"))
                        if case.get("incident_date") else None,
                    documents=repo.list_documents(), uploaded_evidence=case.get("evidence", []),
                    embedding_model=settings.rag_embedding_model,
                    method=os.getenv("BLACKWING_AGENT_RAG_METHOD", "hybrid").strip().lower(),
                    limit_per_agent=3,
                )
        except Exception:
            from blackwing.rag.agent_router import AGENT_ROLES, _case_uploads
            uploads = _case_uploads(case.get("evidence", []), case.get("source_incident_id") or case["case_id"])
            agent_evidence = {role: ([] if role == "similar_case" else uploads) for role in AGENT_ROLES}
    record_event(lifecycle_event("workflow_execution", "invoking_workflow", case_id=case["case_id"]))
    result = await runner(
        case["narrative"],
        settings,
        case_id=case["case_id"],
        source_incident_id=case.get("source_incident_id"),
        evidence=evidence,
        retrieval_count=len(retrieved),
        jurisdiction=case.get("jurisdiction"),
        incident_date=str(case.get("incident_date")) if case.get("incident_date") else None,
        **({"on_update": checkpoint, "on_tool_event": record_event, "on_execution_event": record_event} if runner is analyze_incident else {}),
        **({"agent_evidence": agent_evidence} if agent_evidence is not None else {}),
    )
    existing_trace = result.get("trace", [])
    seen = {e.get("invocation_id") for e in existing_trace if e.get("invocation_id")}
    extra_tools = [e for e in invocation_events
                   if e.get("tool") and e.get("invocation_id") not in seen]
    result["trace"] = [*extra_tools, *existing_trace]
    events = []
    for sequence, event in enumerate(result.get("trace", []), 1):
        events.append({
            **{key: event[key] for key in ("tool", "invocation_id", "case_id", "input_identifiers", "result_summary", "duration_ms", "error", "event_id", "execution_id", "agent_key", "provider", "request_id", "attempt") if key in event},
            "sequence": sequence,
            "step": event.get("step", "unknown"),
            "agent": event.get("agent"),
            "status": event.get("status", "complete"),
            "model": event.get("model", result.get("model")),
            "detail": str(event.get("detail") or event.get("retrieved_document_count")) if event.get("detail") or event.get("retrieved_document_count") is not None else None,
            "timestamp": datetime.fromisoformat(event["timestamp"]) if event.get("timestamp") else datetime.now(timezone.utc),
        })
    repo.patch_run(run_id, {
        "status": "completed",
        "model": result.get("model", settings.model_name),
        "progress": [{"step": event["step"], "status": event["status"]} for event in result.get("trace", [])],
        "trace": events,
        "result": result,
        "report_references": {"run_id": run_id, "json": f"/api/v1/cases/{case['case_id']}/report/export?format=json", "markdown": f"/api/v1/cases/{case['case_id']}/report/export?format=md"},
        "completed_at": datetime.now(timezone.utc),
    })
    repo.update_case(case["case_id"], {"status": "analysis_complete"})
    record_event(lifecycle_event("workflow_execution", "execution_completed", case_id=case["case_id"]))


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "blackwing-api"}


@app.get("/api/v1/capabilities", tags=["system"])
async def capabilities() -> dict[str, Any]:
    return {
        "api_version": "v1",
        "incident_categories": [{"id": "runway_incursion", "status": "supported"}],
        "workflow": "runway_incursion_investigation",
        "agent_roles": ["investigation", "similar_case", "compliance", "risk", "mitigation", "critic", "verification", "report"],
        "features": {
            "case_evidence": True,
            "knowledge_document_ingestion": True,
            "keyword_retrieval": True,
            "hybrid_retrieval": True,
            "seeded_reference_cases": True,
            "risk_matrix_scoring": True,
            "human_review_recording": True,
        },
    }


@app.post("/api/v1/cases", response_model=CaseResponse, status_code=status.HTTP_201_CREATED, tags=["cases"])
async def create_case(
    request: CaseCreate,
    repo: MemoryRepository = Depends(get_repository),
) -> CaseResponse:
    case_id = str(uuid4())
    payload = request.model_dump(mode="json")
    evidence = []
    for item in payload.pop("evidence", []):
        evidence.append({**item, "evidence_id": str(uuid4()), "created_at": datetime.now(timezone.utc).isoformat()})
    record = repo.create_case(case_id, {**payload, "evidence": evidence})
    return CaseResponse.model_validate(record)


@app.get("/api/v1/cases", response_model=CaseListResponse, tags=["cases"])
async def list_cases(
    case_status: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    repo: MemoryRepository = Depends(get_repository),
) -> CaseListResponse:
    items, total = repo.list_cases(case_status, limit, offset)
    return CaseListResponse(
        items=[CaseResponse.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get("/api/v1/cases/{case_id}", response_model=CaseResponse, tags=["cases"])
async def get_case(case_id: str, repo: MemoryRepository = Depends(get_repository)) -> CaseResponse:
    return CaseResponse.model_validate(_case_or_404(repo, case_id))


@app.patch("/api/v1/cases/{case_id}", response_model=CaseResponse, tags=["cases"])
async def update_case(
    case_id: str,
    request: CasePatch,
    repo: MemoryRepository = Depends(get_repository),
) -> CaseResponse:
    case = _case_or_404(repo, case_id)
    if case["status"] != "draft":
        raise HTTPException(status_code=409, detail="Cases can only be edited before the first analysis run")
    fields = request.model_dump(mode="json", exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=422, detail="At least one case field must be supplied")
    updated = repo.update_case(case_id, fields)
    return CaseResponse.model_validate(updated)


@app.post("/api/v1/cases/{case_id}/evidence", response_model=EvidenceResponse, status_code=status.HTTP_201_CREATED, tags=["cases"])
async def add_case_evidence(
    case_id: str,
    request: EvidenceInput,
    repo: MemoryRepository = Depends(get_repository),
) -> EvidenceResponse:
    case = _case_or_404(repo, case_id)
    if case["status"] != "draft":
        raise HTTPException(status_code=409, detail="Evidence cannot be changed after analysis starts; create a new case revision")
    evidence = request.model_dump(mode="json")
    record = repo.add_evidence(case_id, {
        **evidence,
        "evidence_id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return EvidenceResponse.model_validate(record)


@app.post("/api/v1/cases/{case_id}/runs", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED, tags=["analysis"])
async def create_analysis_run(
    case_id: str,
    request: RunCreate,
    background: BackgroundTasks,
    repo: MemoryRepository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
    runner: AnalysisRunner = Depends(get_analysis_runner),
) -> RunResponse:
    case = _case_or_404(repo, case_id)
    if case["status"] == "analysis_running":
        raise HTTPException(status_code=409, detail="An analysis run is already active for this case")
    query = " ".join(part for part in [case["narrative"], case.get("location"), "runway incursion"] if part)
    run_id = str(uuid4())
    run = repo.create_run(run_id, case_id, request.workflow, settings.model_name)
    preflight = []
    def record(event):
        preflight.append(event)
        repo.append_event(run_id, event)
    with tool_audit(case_id, record):
        try:
            retrieved = retrieve_evidence(
                retriever.search, input_identifiers=[case.get("source_incident_id") or case_id],
                query=query, documents=repo.list_documents(), embedding_model=settings.rag_embedding_model,
                document_types=["incident_report", "regulation", "procedure", "guidance"],
                jurisdiction=case.get("jurisdiction"),
                incident_date=str(case.get("incident_date")) if case.get("incident_date") else None,
                limit=5,
            )["results"] if request.include_retrieval else []
        except Exception:
            # Continue with uploads, disclose retrieval failure, never invent sources.
            retrieved = []
    repo.patch_run(run_id, {"preflight_trace": preflight})
    background.add_task(_execute_run, run_id, case, retrieved, settings, repo, runner, request.include_retrieval)
    return _run_response(run)


@app.get("/api/v1/runs/{run_id}", response_model=RunResponse, tags=["analysis"])
async def get_run(run_id: str, repo: MemoryRepository = Depends(get_repository)) -> RunResponse:
    run = repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_response(run)


@app.get("/api/v1/cases/{case_id}/runs", response_model=list[RunResponse], tags=["analysis"])
async def list_case_runs(case_id: str, repo: MemoryRepository = Depends(get_repository)) -> list[RunResponse]:
    _case_or_404(repo, case_id)
    return [_run_response(run) for run in repo.list_case_runs(case_id)]


@app.get("/api/v1/runs/{run_id}/trace", response_model=TraceResponse, tags=["analysis"])
async def get_trace(run_id: str, repo: MemoryRepository = Depends(get_repository)) -> TraceResponse:
    run = repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    events = list(run.get("trace", []))
    known_events = {e.get("event_id") for e in events if e.get("event_id")}
    known_tools = {e.get("invocation_id") for e in events if e.get("invocation_id")}
    for event in run.get("audit_events", []):
        if event.get("event_id") and event["event_id"] not in known_events or event.get("tool") and event.get("invocation_id") not in known_tools:
            copy = dict(event)
            if copy.get("detail") is not None and not isinstance(copy["detail"], str):
                copy["detail"] = json.dumps(copy["detail"], ensure_ascii=False)
            events.append(copy)
    def event_time(event):
        value = event.get("timestamp") or run["created_at"]
        return datetime.fromisoformat(value) if isinstance(value, str) else value
    events.sort(key=event_time)
    return TraceResponse(run_id=run_id, events=[{**event, "sequence": index} for index, event in enumerate(events, 1)])


@app.get("/api/v1/cases/{case_id}/report", response_model=ReportResponse, tags=["reports"])
async def get_report(case_id: str, repo: MemoryRepository = Depends(get_repository)) -> ReportResponse:
    case = _case_or_404(repo, case_id)
    runs = repo.list_case_runs(case_id)
    completed = [run for run in runs if run["status"] == "completed"]
    if not completed:
        latest = max(runs, key=lambda run: run["created_at"]) if runs else None
        if latest and latest.get("error_code") == "interrupted":
            detail = "Report unavailable: execution was interrupted. Trace history is preserved; start a new recovery run."
        elif latest and latest["status"] in {"queued", "running"}:
            detail = "Report unavailable: investigation is still running. Refresh run status and inspect the execution trace."
        elif latest and latest["status"] == "failed":
            detail = "Report unavailable: investigation failed. Inspect the execution trace before starting a new run."
        else:
            detail = "Report unavailable: no investigation has completed for this case."
        raise HTTPException(status_code=409, detail=detail)
    latest = max(completed, key=lambda run: run["created_at"])
    return ReportResponse.model_validate({
        "case_id": case_id,
        "run_id": latest["run_id"],
        "case_status": case["status"],
        "analysis_status": latest["status"],
        "report": latest["result"],
        "reviews": repo.list_reviews(case_id),
        "is_final": False,
        "notice": "AI output is preliminary until a qualified human reviewer records a decision.",
    })


@app.get("/api/v1/cases/{case_id}/report/export", tags=["reports"])
async def export_report(
    case_id: str,
    export_format: str = Query(default="json", alias="format", pattern="^(json|md)$"),
    repo: MemoryRepository = Depends(get_repository),
) -> Response:
    report = await get_report(case_id, repo)
    if export_format == "md":
        from blackwing.reporting import render_markdown_report
        filename = f"blackwing-{case_id}-report.md"
        return Response(
            content=render_markdown_report(report.model_dump(mode="json")),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    filename = f"blackwing-{case_id}-report.json"
    return Response(
        content=json.dumps(report.model_dump(mode="json"), ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/v1/cases/{case_id}/reviews", response_model=ReviewResponse, status_code=status.HTTP_201_CREATED, tags=["review"])
async def create_review(
    case_id: str,
    request: ReviewCreate,
    repo: MemoryRepository = Depends(get_repository),
) -> ReviewResponse:
    _case_or_404(repo, case_id)
    if not any(run["case_id"] == case_id and run["status"] == "completed" for run in repo.list_case_runs(case_id)):
        raise HTTPException(status_code=409, detail="A completed analysis run is required before review")
    record = {
        **request.model_dump(mode="json"),
        "review_id": str(uuid4()),
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return ReviewResponse.model_validate(repo.add_review(case_id, record))


@app.get("/api/v1/cases/{case_id}/reviews", response_model=list[ReviewResponse], tags=["review"])
async def list_reviews(case_id: str, repo: MemoryRepository = Depends(get_repository)) -> list[ReviewResponse]:
    _case_or_404(repo, case_id)
    return [ReviewResponse.model_validate(row) for row in repo.list_reviews(case_id)]


@app.post("/api/v1/knowledge/documents", response_model=KnowledgeDocumentResponse, status_code=status.HTTP_201_CREATED, tags=["knowledge"])
async def ingest_knowledge_document(
    request: KnowledgeDocumentCreate,
    repo: MemoryRepository = Depends(get_repository),
) -> KnowledgeDocumentResponse:
    record = repo.add_document(str(uuid4()), request.model_dump(mode="json"))
    return KnowledgeDocumentResponse(
        **{key: record.get(key) for key in (
            "document_id", "title", "document_type", "source_url", "source_id",
            "authority_type", "source_status", "version", "jurisdiction", "document_date",
            "effective_date", "page_number", "section_reference", "created_at",
        )},
        content_length=len(record["content"]),
    )


@app.get("/api/v1/knowledge/documents", response_model=list[KnowledgeDocumentResponse], tags=["knowledge"])
async def list_knowledge_documents(
    document_type: str | None = Query(default=None),
    jurisdiction: str | None = Query(default=None),
    repo: MemoryRepository = Depends(get_repository),
) -> list[KnowledgeDocumentResponse]:
    return [
        KnowledgeDocumentResponse(
            **{key: row.get(key) for key in (
                "document_id", "title", "document_type", "source_url", "source_id",
                "authority_type", "source_status", "version", "jurisdiction", "document_date",
                "effective_date", "page_number", "section_reference", "created_at",
            )},
            content_length=len(row["content"]),
        )
        for row in repo.list_documents(document_type, jurisdiction)
    ]


@app.get("/api/v1/knowledge/documents/{document_id}", response_model=KnowledgeDocumentResponse, tags=["knowledge"])
async def get_knowledge_document(document_id: str, repo: MemoryRepository = Depends(get_repository)) -> KnowledgeDocumentResponse:
    row = repo.get_document(document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Knowledge document not found")
    return KnowledgeDocumentResponse(
        **{key: row.get(key) for key in (
            "document_id", "title", "document_type", "source_url", "source_id",
            "authority_type", "source_status", "version", "jurisdiction", "document_date",
            "effective_date", "page_number", "section_reference", "created_at",
        )},
        content_length=len(row["content"]),
    )


@app.post("/api/v1/knowledge/search", response_model=KnowledgeSearchResponse, tags=["knowledge"])
async def search_knowledge(
    request: KnowledgeSearchRequest,
    repo: MemoryRepository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
) -> KnowledgeSearchResponse:
    search_result = retriever.search(
        request.query,
        repo.list_documents(),
        embedding_model=settings.rag_embedding_model,
        document_types=request.document_types,
        jurisdiction=request.jurisdiction,
        incident_date=request.incident_date.isoformat() if request.incident_date else None,
        limit=request.limit,
    )
    return KnowledgeSearchResponse(
        results=search_result["results"],
        retrieval_method=search_result["retrieval_method"],
        limitations=search_result["limitations"],
    )


@app.get("/api/v1/runs/{run_id}/audit", tags=["analysis"])
async def get_run_audit(run_id: str, repo: MemoryRepository = Depends(get_repository)) -> dict[str, Any]:
    run = repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run_id": run_id, "events": run.get("audit_events", []), "plan": run.get("plan"),
            "partial_outputs": run.get("partial_outputs", {}), "execution_steps": run.get("execution_steps", 0),
            "execution_state": run.get("execution_state"), "resumable": False, "recovery": "Create a new run after interruption; prior run history is retained."}
