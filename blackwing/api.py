from __future__ import annotations

from datetime import datetime, timezone
import json
import hashlib
import asyncio
import logging
import os
from time import perf_counter
from typing import Any, Awaitable, Callable
from uuid import uuid4

import httpx

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
    OperationsMetricsResponse,
    ReportResponse,
    ReviewCreate,
    ReviewResponse,
    RunCreate,
    RunResponse,
    TraceResponse,
    RunAuditResponse,
)
from blackwing.agents.base import AgentExecutionError
from blackwing.service import analyze_incident
from blackwing.rag.retriever import retriever
from blackwing.tools import retrieve_evidence, tool_audit
from blackwing.diagnostics import lifecycle_event, safe_error
from blackwing.incident_identity import resolve_incident_reference, declared_intake_metadata
from blackwing.observability import start_run as start_external_trace, finish_run as finish_external_trace

AnalysisRunner = Callable[..., Awaitable[dict[str, Any]]]
logger = logging.getLogger("blackwing.runs")

app = FastAPI(
    title="BlackWing AI API",
    description=(
        "Versioned API for BlackWing runway-incursion investigations. Create cases and evidence, "
        "start and inspect multi-agent runs, retrieve reports, record human review, and inspect "
        "run-level trace, audit, and aggregate operations metrics. Hosted model and embedding "
        "credentials are configured on the API server and are never accepted in request bodies."
    ),
    version="1.0.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_tags=[
        {"name": "system", "description": "Health checks and supported API/workflow capabilities."},
        {"name": "cases", "description": "Create and inspect incident cases and attach case evidence."},
        {"name": "analysis", "description": "Start, resume, inspect, and trace multi-agent investigation runs."},
        {"name": "reports", "description": "Read and export published reports; evaluator-only comparisons are post-generation."},
        {"name": "evaluation", "description": "Aggregate run telemetry and evaluation status. Accuracy is reported only when judged."},
        {"name": "review", "description": "Record human review decisions and finding overrides."},
        {"name": "knowledge", "description": "Manage knowledge documents and search the configured RAG index."},
        {"name": "final report evaluation", "description": "Compare published report sections with an evaluator-only gold PDF; gold material is excluded from generation and RAG."},
    ],
)


def get_repository() -> MemoryRepository:
    return repository


def get_settings() -> Settings:
    try:
        settings = Settings.load()
        retriever.configure_index_dir(settings.rag_index_dir)
        retriever.configure_embedding_settings(settings)
        return settings
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
    retrieved: list[dict[str, Any]] | None,
    settings: Settings,
    repo: MemoryRepository,
    runner: AnalysisRunner,
    use_agent_retrieval: bool = True,
    resume: bool = False,
) -> None:
    """Contain background failures; cooperative server shutdown still propagates."""
    started = perf_counter()
    trace = start_external_trace(settings, run_id=run_id, case_id=case.get("case_id", ""))
    status = "failed"
    failure_type = None
    try:
        await _execute_run_body(run_id, case, retrieved, settings, repo, runner, use_agent_retrieval, resume)
        saved = repo.get_run(run_id) or {}
        status = saved.get("status", "completed")
        failure_type = saved.get("error_code")
    except asyncio.CancelledError as exc:
        failure_type = type(exc).__name__
        _record_run_failure(run_id, case, repo, exc, interrupted=True)
        raise
    except (Exception, SystemExit) as exc:
        failure_type = type(exc).__name__
        _record_run_failure(run_id, case, repo, exc)
    finally:
        saved = repo.get_run(run_id) or {}
        audit = saved.get("audit_events", [])
        tool_calls = sum(1 for event in audit if event.get("tool"))
        prompt_tokens = sum(int(event.get("prompt_tokens") or 0) for event in audit)
        completion_tokens = sum(int(event.get("completion_tokens") or 0) for event in audit)
        cost = ((prompt_tokens * settings.prompt_cost_per_million +
                 completion_tokens * settings.completion_cost_per_million) / 1_000_000
                if settings.prompt_cost_per_million is not None and
                   settings.completion_cost_per_million is not None else None)
        try:
            repo.patch_run(run_id, {"operations_metrics": {
                "latency_ms": round((perf_counter() - started) * 1000, 1),
                "tool_calls": tool_calls, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, "estimated_cost_usd": cost,
            }})
        except Exception:
            logger.warning("Could not persist operations metrics; run=%s", run_id)
        finish_external_trace(trace, status=status, elapsed_ms=(perf_counter() - started) * 1000,
                              tool_calls=tool_calls, failure_type=failure_type)


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
    retrieved: list[dict[str, Any]] | None,
    settings: Settings,
    repo: MemoryRepository,
    runner: AnalysisRunner,
    use_agent_retrieval: bool = True,
    resume: bool = False,
) -> None:
    case = {**case, "source_incident_id": resolve_incident_reference(case["narrative"], case.get("source_incident_id"))}
    repo.patch_run(run_id, {"status": "running", "started_at": datetime.now(timezone.utc)})
    stored = repo.get_run(run_id) if hasattr(repo, "get_run") else {}
    invocation_events = list((stored or {}).get("preflight_trace", []))
    def record_event(event):
        invocation_events.append(event)
        if hasattr(repo, "append_event"):
            repo.append_event(run_id, event)
        else:
            repo.patch_run(run_id, {"audit_events": list(invocation_events)})
    def checkpoint(state):
        fields = {"partial_outputs": state.get("outputs", {}), "plan": state.get("plan"),
                               "execution_steps": state.get("execution_steps", 0),
                               "progress": [{"step": e["step"], "status": e["status"]} for e in state.get("trace", [])] + [e for e in invocation_events if e.get("event_id")]}
        if "evidence" in state and not (repo.get_run(run_id) or {}).get("evaluation_input_snapshot"):
            from blackwing.agents.report.provenance import collect_report_evidence
            rows = collect_report_evidence(state.get("evidence", []), state.get("agent_evidence"))
            fields["evaluation_input_snapshot"] = {"version": 1, "complete": True,
                "scope": "Workflow evidence before per-agent token clipping; not a reconstruction of each model message.",
                "evidence": rows, "sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode('utf-8')).hexdigest()}
        repo.patch_run(run_id, fields)
        record_event({"step": "state_update", "status": "complete", "detail": state.get("trace", [])[-1:]})
    record_event(lifecycle_event("workflow_execution", "execution_started", case_id=case["case_id"],
        detail="Retrieval incident reference: " + str(case.get("source_incident_id") or "unspecified")))
    saved_resume = (repo.get_run(run_id) or {}).get("resume_payload") if resume else None
    if saved_resume:
        retrieved = saved_resume.get("retrieved", [])
    if retrieved is None:
        record_event(lifecycle_event("evidence_retrieval", "retrieval_started", case_id=case["case_id"],
                                    detail="Searching the existing evidence index"))
        def initial_retrieval():
            from blackwing.rag.query_text import case_query_text
            query = " ".join(part for part in [case_query_text(case["narrative"]), case.get("location"), "runway incursion"] if part)
            with tool_audit(case["case_id"], record_event):
                return retrieve_evidence(retriever.search,
                    input_identifiers=[case.get("source_incident_id") or case["case_id"]],
                    query=query, documents=repo.list_documents(), embedding_model=settings.rag_embedding_model,
                    document_types=["incident_report", "regulation", "procedure", "guidance"],
                    jurisdiction=case.get("jurisdiction"),
                    incident_date=str(case.get("incident_date")) if case.get("incident_date") else None,
                    limit=5)["results"]
        try:
            retrieved = await asyncio.to_thread(initial_retrieval)
            status = "retrieval_completed"
        except Exception:
            retrieved = []
            status = "retrieval_failed"
        repo.patch_run(run_id, {"preflight_trace": [e for e in invocation_events if e.get("tool")]})
        record_event(lifecycle_event("evidence_retrieval", status, case_id=case["case_id"],
                                     retrieved_document_count=len(retrieved)))
    from blackwing.rag.agent_router import _case_uploads
    uploads = _case_uploads(case.get("evidence", []), case.get("source_incident_id") or case["case_id"])
    evidence = [*uploads, *retrieved]
    agent_evidence = saved_resume.get("agent_evidence") if saved_resume else None
    if not resume and use_agent_retrieval and os.getenv("BLACKWING_AGENT_RAG", "0") == "1":
        # Keep integration outside agent packages and UI. No LLM calls here.
        from blackwing.rag.agent_router import build_agent_packets
        record_event(lifecycle_event("evidence_retrieval", "retrieval_started", case_id=case["case_id"],
                                    detail="Preparing agent-specific evidence packets"))
        def agent_retrieval():
            with tool_audit(case["case_id"], record_event):
                return retrieve_evidence(build_agent_packets, input_identifiers=[case["case_id"]],
                    retriever=retriever, narrative=case["narrative"],
                    case_id=case.get("source_incident_id") or case["case_id"],
                    jurisdiction=case.get("jurisdiction"), incident_date=str(case.get("incident_date"))
                        if case.get("incident_date") else None,
                    documents=repo.list_documents(), uploaded_evidence=case.get("evidence", []),
                    embedding_model=settings.rag_embedding_model,
                    method=os.getenv("BLACKWING_AGENT_RAG_METHOD", "hybrid").strip().lower(),
                    limit_per_agent=5,
                )
        try:
            agent_evidence = await asyncio.to_thread(agent_retrieval)
            record_event(lifecycle_event("evidence_retrieval", "retrieval_completed", case_id=case["case_id"]))
        except Exception:
            from blackwing.rag.agent_router import AGENT_ROLES, _case_uploads
            uploads = _case_uploads(case.get("evidence", []), case.get("source_incident_id") or case["case_id"])
            agent_evidence = {role: ([] if role == "similar_case" else uploads) for role in AGENT_ROLES}
    record_event(lifecycle_event("workflow_execution", "invoking_workflow", case_id=case["case_id"]))
    if not resume:
        repo.patch_run(run_id, {"resume_payload": {"retrieved": retrieved or [],
            "agent_evidence": agent_evidence, "checkpoint_thread_id": run_id}})
    result = await runner(
        case["narrative"],
        settings,
        case_id=case["case_id"],
        source_incident_id=case.get("source_incident_id"),
        evidence=evidence,
        retrieval_count=len(retrieved),
        jurisdiction=case.get("jurisdiction"),
        incident_date=str(case.get("incident_date")) if case.get("incident_date") else None,
        **({"on_update": checkpoint, "on_tool_event": record_event, "on_execution_event": record_event,
            "checkpoint_thread_id": run_id, "resume_from_checkpoint": resume} if runner is analyze_incident else {}),
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


@app.get("/api/v1/metrics/operations", response_model=OperationsMetricsResponse, tags=["evaluation"],
         summary="Get aggregate operations metrics",
         description="Returns task completion, latency, tool-call, token, estimated-cost, and failure-taxonomy metrics across saved investigation runs. End-to-end accuracy is null unless a completed evaluation supplied a score. Estimated costs require both configured token rates.")
async def operations_metrics(repo: MemoryRepository = Depends(get_repository),
                             settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Aggregate durable run telemetry; accuracy is reported only when judged."""
    runs = [run for case in repo.list_cases(status=None, limit=100000, offset=0)[0]
            for run in repo.list_case_runs(case["case_id"])]
    total = len(runs)
    completed = sum(run.get("status") == "completed" for run in runs)
    failed = [run for run in runs if run.get("status") == "failed"]
    metrics = [run.get("operations_metrics", {}) for run in runs if run.get("operations_metrics")]
    valid_latency = [m["latency_ms"] for m in metrics if isinstance(m.get("latency_ms"), (int, float))]
    tool_counts = [m.get("tool_calls", 0) for m in metrics]
    prompt_tokens = sum(m.get("prompt_tokens", 0) for m in metrics)
    completion_tokens = sum(m.get("completion_tokens", 0) for m in metrics)
    cost_values = [m["estimated_cost_usd"] for m in metrics if isinstance(m.get("estimated_cost_usd"), (int, float))]
    judged_scores = [run.get("question_evaluation", {}).get("score") for run in runs
                     if run.get("question_evaluation", {}).get("status") == "completed"
                     and isinstance(run.get("question_evaluation", {}).get("score"), (int, float))]
    taxonomy = {"retrieval_miss": 0, "tool_call_error": 0, "reasoning_error": 0,
                "hallucination": None, "other": 0}
    for run in failed:
        events = run.get("audit_events", [])
        if any(e.get("step") == "evidence_retrieval" and (e.get("status") in {"retrieval_failed", "retrieval_no_matches"}
               or (e.get("status") == "retrieval_completed" and e.get("retrieved_document_count") == 0)) for e in events):
            taxonomy["retrieval_miss"] += 1
        elif any(e.get("tool") and e.get("error") for e in events):
            taxonomy["tool_call_error"] += 1
        elif run.get("error_code") in {"invalid_agent_output", "agent_processing_error", "context_budget_exceeded"}:
            taxonomy["reasoning_error"] += 1
        else:
            taxonomy["other"] += 1
    # Hallucination requires an explicit groundedness judgment; do not infer it
    # from a failed run.
    return {
        "run_count": total,
        "task_completion_rate": completed / total if total else None,
        "end_to_end_accuracy": sum(judged_scores) / len(judged_scores) if judged_scores else None,
        "end_to_end_accuracy_basis": "Mean saved gold-question report score (0-100); not the Week 1 benchmark." if judged_scores else "No completed gold-question evaluation is saved.",
        "average_tool_calls_per_run": sum(tool_counts) / len(tool_counts) if tool_counts else None,
        "average_tool_calls_per_task": sum(tool_counts) / len(tool_counts) if tool_counts else None,
        "average_latency_ms": sum(valid_latency) / len(valid_latency) if valid_latency else None,
        "average_latency_ms_per_query": sum(valid_latency) / len(valid_latency) if valid_latency else None,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "estimated_cost_usd": sum(cost_values) if cost_values else None,
        "average_cost_usd_per_query": sum(cost_values) / len(cost_values) if cost_values else None,
        "cost_note": "Estimated only when both per-million token rates are configured.",
        "failure_taxonomy": taxonomy,
        "hallucination_note": "Requires claim-level evidence evaluation; never inferred from status alone.",
        "cost_rates_configured": settings.prompt_cost_per_million is not None and settings.completion_cost_per_million is not None,
    }


@app.post("/api/v1/cases", response_model=CaseResponse, status_code=status.HTTP_201_CREATED, tags=["cases"])
async def create_case(
    request: CaseCreate,
    repo: MemoryRepository = Depends(get_repository),
) -> CaseResponse:
    case_id = str(uuid4())
    payload = request.model_dump(mode="json")
    for key, value in declared_intake_metadata(payload['narrative']).items():
        if payload.get(key) is None:
            payload[key] = value
    payload["source_incident_id"] = resolve_incident_reference(payload["narrative"], payload.get("source_incident_id"))
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
    run_id = str(uuid4())
    run = repo.create_run(run_id, case_id, request.workflow, settings.model_name)
    # Return 202 before retrieval or inference. Their synchronous adapters run
    # in worker threads so health, status, trace and UI polling remain responsive.
    background.add_task(_execute_run, run_id, case, None if request.include_retrieval else [],
                        settings, repo, runner, request.include_retrieval)
    return _run_response(run)


@app.get("/api/v1/runs/{run_id}", response_model=RunResponse, tags=["analysis"])
async def get_run(run_id: str, repo: MemoryRepository = Depends(get_repository)) -> RunResponse:
    run = repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_response(run)


@app.post("/api/v1/runs/{run_id}/resume", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED, tags=["analysis"])
async def resume_analysis_run(run_id: str, background: BackgroundTasks,
                              repo: MemoryRepository = Depends(get_repository),
                              settings: Settings = Depends(get_settings),
                              runner: AnalysisRunner = Depends(get_analysis_runner)) -> RunResponse:
    run = repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.get("status") != "failed" or run.get("error_code") != "interrupted" or not run.get("resume_payload"):
        raise HTTPException(status_code=409, detail="This run has no resumable checkpoint. Start a new analysis run.")
    case = _case_or_404(repo, run["case_id"])
    if case.get("status") == "analysis_running":
        raise HTTPException(status_code=409, detail="Another analysis run is active for this case")
    repo.patch_run(run_id, {"status": "queued", "error_code": None, "error": None,
                            "started_at": datetime.now(timezone.utc), "completed_at": None,
                            "resumable": False})
    repo.update_case(case["case_id"], {"status": "analysis_running"})
    background.add_task(_execute_run, run_id, case, None, settings, repo, runner, True, True)
    return _run_response(repo.get_run(run_id))


@app.get("/api/v1/cases/{case_id}/runs", response_model=list[RunResponse], tags=["analysis"])
async def list_case_runs(case_id: str, repo: MemoryRepository = Depends(get_repository)) -> list[RunResponse]:
    _case_or_404(repo, case_id)
    return [_run_response(run) for run in repo.list_case_runs(case_id)]


@app.get("/api/v1/runs/{run_id}/trace", response_model=TraceResponse, tags=["analysis"],
         summary="Get a chronological run trace",
         description="Returns ordered workflow, retrieval, agent, model-request, and tool lifecycle events for one run. This view omits per-request token counters; use the audit endpoint for detailed persisted telemetry.",
         responses={404: {"description": "Run not found"}})
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
            # Token usage belongs to the private operations metrics/audit store,
            # not the strict public trace response schema.
            copy.pop("prompt_tokens", None)
            copy.pop("completion_tokens", None)
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
    report_result = dict(latest["result"])
    if isinstance(report_result.get("trace"), list):
        report_result["trace"] = [
            {key: value for key, value in event.items()
             if key not in {"prompt_tokens", "completion_tokens"}}
            if isinstance(event, dict) else event
            for event in report_result["trace"]
        ]
    return ReportResponse.model_validate({
        "case_id": case_id,
        "run_id": latest["run_id"],
        "case_status": case["status"],
        "analysis_status": latest["status"],
        "report": report_result,
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
    from blackwing.response_cache import knowledge_search_cache
    documents = repo.list_documents()
    cache_key = knowledge_search_cache.key(request.model_dump(mode="json"), documents,
                                          str(retriever.index_dir) if retriever.index_dir else None,
                                          settings.rag_embedding_model)
    search_result = knowledge_search_cache.get(cache_key, settings.knowledge_cache_ttl_seconds)
    if search_result is None:
        search_result = retriever.search(
        request.query,
        documents,
        embedding_model=settings.rag_embedding_model,
        document_types=request.document_types,
        jurisdiction=request.jurisdiction,
        incident_date=request.incident_date.isoformat() if request.incident_date else None,
        limit=request.limit,
        )
        knowledge_search_cache.set(cache_key, search_result, settings.knowledge_cache_ttl_seconds)
    return KnowledgeSearchResponse(
        results=search_result["results"],
        retrieval_method=search_result["retrieval_method"],
        limitations=search_result["limitations"],
    )


@app.get("/api/v1/runs/{run_id}/audit", response_model=RunAuditResponse, tags=["analysis"],
         summary="Get run audit and operations details",
         description="Returns persisted run events, workflow plan, partial outputs, execution state, lifecycle timestamps, and aggregate token/tool/latency/cost metrics. Event details may include case evidence or agent work products; do not publish this response to untrusted clients.",
         responses={404: {"description": "Run not found"}})
async def get_run_audit(run_id: str, repo: MemoryRepository = Depends(get_repository)) -> dict[str, Any]:
    run = repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run_id": run_id, "events": run.get("audit_events", []), "plan": run.get("plan"),
            "partial_outputs": run.get("partial_outputs", {}), "execution_steps": run.get("execution_steps", 0),
            "execution_state": run.get("execution_state"), "operations_metrics": run.get("operations_metrics", {}),
            "status": run.get("status"), "workflow": run.get("workflow"), "model": run.get("model"),
            "created_at": run.get("created_at"), "started_at": run.get("started_at"),
            "completed_at": run.get("completed_at"), "resumable": False,
            "recovery": "Create a new run after interruption; prior run history is retained."}


# Evaluation-only routes: gold never enters case evidence or generation.
_gold_locks: dict[str, asyncio.Lock] = {}


def _gold_context(case_id, run_id, repo):
    from blackwing.gold_comparison import match_gold, report_hash, VERSION, gold_path, comparison_progress
    case = _case_or_404(repo, case_id)
    run = repo.get_run(run_id)
    if not run or run['case_id'] != case_id:
        raise HTTPException(status_code=404, detail='Report run not found')
    if run['status'] != 'completed' or not run.get('result'):
        raise HTTPException(status_code=409, detail='A completed report is required for gold comparison.')
    from pydantic import ValidationError
    try:
        report = ReportResponse.model_validate({
            'case_id':case_id,'run_id':run_id,'case_status':case['status'],
            'analysis_status':'completed','report':run['result'],'reviews':[],'is_final':False,
            'notice':'Human review required.'}).report.model_dump(mode='json')
    except ValidationError:
        item = match_gold(case)
        context = {'status':'not_assessable','score':None,'label':'Gold comparison score',
            'error_code':'legacy_report_schema_incompatible', 'reference_case':bool(item),
            'reason':'This saved report does not satisfy the current backend schema. '
                     'Its risk values and history have not been converted. Create a new investigation '
                     'with the current backend before comparison; the original report is preserved.'}
        if item:
            context.update(gold_id=item['gold_id'],gold_case=item['case_label'])
        return case, run, {}, context
    from blackwing.case_inputs import is_synthetic_case
    if is_synthetic_case(case.get('narrative')):
        from blackwing.synthetic_gold import context as synthetic_context
        context = synthetic_context(case, report, run)
        if run.get('gold_comparison_error') and context.get('status') != 'proposed':
            context['last_error'] = run['gold_comparison_error']
        return case, run, report, context
    item = match_gold(case)
    context = {'status':'not_applicable','score':None,'label':'Gold comparison score',
               'reason':'No uniquely matching preserved gold case; fictional and unrelated cases are not compared.'}
    if item:
        context.update(status='not_requested',gold_id=item['gold_id'],gold_case=item['case_label'],
                       reason='Select Compare with gold report to request a comparison.')
        try:
            gold_path(item)
        except ValueError as exc:
            context.update(status='unavailable',reason=str(exc))
        cached = run.get('gold_comparison') or {}
        if (context['status'] != 'unavailable' and cached.get('report_hash') == report_hash(report)
                and cached.get('gold_sha256') == item['sha256'] and cached.get('rubric_version') == VERSION):
            context = comparison_progress(cached)
    if run.get('gold_comparison_error') and context.get('status') not in {'proposed','not_applicable'}:
        context['last_error'] = run['gold_comparison_error']
    return case, run, report, context


@app.get('/api/v1/cases/{case_id}/runs/{run_id}/gold-comparison', tags=['reports'])
async def get_gold_comparison(case_id: str, run_id: str, repo: MemoryRepository = Depends(get_repository)):
    return _gold_context(case_id, run_id, repo)[3]


@app.post('/api/v1/cases/{case_id}/runs/{run_id}/gold-comparison', tags=['reports'])
async def create_gold_comparison(case_id: str, run_id: str,
        repo: MemoryRepository = Depends(get_repository), settings: Settings = Depends(get_settings)):
    from blackwing.gold_comparison import compare_report
    async with _gold_locks.setdefault(run_id, asyncio.Lock()):
        case, run, report, cached = _gold_context(case_id, run_id, repo)
        if cached['status'] in {'not_applicable','proposed','not_assessable'}:
            return cached
        if cached['status'] == 'unavailable':
            raise HTTPException(status_code=503, detail=cached['reason'])
        active, _ = repo.list_cases(status='analysis_running', limit=1, offset=0)
        if active:
            raise HTTPException(status_code=409, detail='Wait for active investigations to finish before gold comparison.')
        try:
            if cached['status'] == 'evaluation_incomplete':
                result = await compare_report(case, report, run_id, settings, prior=cached)
            else:
                result = await compare_report(case, report, run_id, settings)
        except Exception as exc:
            from pypdf.errors import DependencyError
            diagnostic = safe_error(exc)
            code = 'gold_comparison_failed'
            message = 'Gold comparison did not complete. No score assigned; inspect the backend diagnostics.'
            if isinstance(exc, DependencyError):
                code = 'gold_pdf_crypto_missing'
                message = 'Gold PDF encryption support is missing. Install the project requirements and restart FastAPI.'
            elif isinstance(exc, httpx.HTTPStatusError):
                code = 'gold_provider_http_error'
                message = f'Gold comparison provider returned HTTP {exc.response.status_code}. No score assigned.'
            elif isinstance(exc, httpx.TimeoutException):
                code = 'gold_provider_timeout'
                message = 'Gold comparison model timed out. No score assigned; retry after the active request finishes.'
            error = {'code':code,'message':message,'error_type':type(exc).__name__}
            repo.patch_run(run_id, {'gold_comparison_error':error})
            logger.error('Gold comparison unavailable; run=%s diagnostic=%s', run_id, json.dumps(diagnostic))
            raise HTTPException(status_code=503, detail=error) from exc
        repo.patch_run(run_id, {'gold_comparison':result,'gold_comparison_error':None})
        return result


from blackwing.qa_routes import build_router as build_qa_router
app.include_router(build_qa_router(get_repository, get_settings))

from blackwing.evaluation_routes import build_router as build_evaluation_router
app.include_router(build_evaluation_router(get_repository, get_settings))

from blackwing.final_report_evaluation_routes import build_router as build_final_report_evaluation_router
app.include_router(build_final_report_evaluation_router(get_repository, get_settings))

from blackwing.retrieval_evaluation_routes import build_router as build_retrieval_evaluation_router
app.include_router(build_retrieval_evaluation_router(get_settings))
