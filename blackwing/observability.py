"""Opt-in LangSmith tracing with metadata-only payloads.

Incident narratives, evidence text, prompts, model outputs, and credentials are
deliberately excluded. LangSmith is disabled by default.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from uuid import UUID

logger = logging.getLogger("blackwing.observability")


def start_run(settings, *, run_id: str, case_id: str):
    if settings.observability_backend != "langsmith":
        return None
    try:
        from langsmith import Client
        client = Client(api_url=settings.langsmith_endpoint, api_key=settings.langsmith_api_key)
        client.create_run(
            name="blackwing_investigation", run_type="chain",
            project_name=settings.langsmith_project,
            id=UUID(run_id), start_time=datetime.now(timezone.utc),
            inputs={"run_id": run_id, "case_id": case_id, "provider": settings.llm_provider,
                    "model": settings.model_name},
            extra={"metadata": {"blackwing_trace_schema": "metadata-only-v1"}},
            tags=["blackwing", "investigation"],
        )
        return client, run_id
    except Exception as exc:
        # Observability must not interrupt an investigation.
        logger.warning("LangSmith trace start failed; type=%s", type(exc).__name__)
        return None


def finish_run(trace, *, status: str, elapsed_ms: float, tool_calls: int,
               failure_type: str | None = None):
    if not trace:
        return
    client, run_id = trace
    try:
        client.update_run(
            run_id, end_time=datetime.now(timezone.utc),
            outputs={"status": status, "latency_ms": round(elapsed_ms, 1),
                     "tool_calls": int(tool_calls)},
            error=failure_type,
        )
    except Exception as exc:
        logger.warning("LangSmith trace update failed; type=%s", type(exc).__name__)
