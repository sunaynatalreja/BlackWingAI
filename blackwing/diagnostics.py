"""Credential-free, investigation-scoped lifecycle diagnostics.

Sinks run synchronously before the model await. A successful sink return means
the local repository transaction has committed, not merely queued an update.
"""
import asyncio
import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import logging
from pathlib import Path
from time import perf_counter
import traceback
from uuid import uuid4

import httpx
from pydantic import ValidationError

from blackwing.agents.base import AgentExecutionError

logger = logging.getLogger("blackwing.execution")
_context = ContextVar("execution_audit", default=None)
_agent = ContextVar("active_agent_execution", default={})

MESSAGES = {
    "model_timeout": "The model request timed out.",
    "model_http_error": "The model provider returned an unsuccessful HTTP status.",
    "model_unreachable": "The model provider could not be reached.",
    "invalid_agent_output": "The agent output failed validation.",
    "context_budget_exceeded": "The agent input exceeded the safe context budget.",
    "agent_processing_error": "The agent failed while processing its output.",
    "internal_agent_error": "An unexpected investigation error occurred.",
    "execution_step_limit": "The investigation execution-step limit was reached.",
    "process_exit_attempt": "Background investigation attempted to exit the process; the exit was contained.",
    "interrupted": "Execution was interrupted; start a new controlled recovery run.",
}


def safe_error(exc, code=None):
    """Allow-list metadata; never copy exception text, bodies, URLs or headers."""
    origin = exc
    seen = set()
    while origin.__cause__ is not None and id(origin) not in seen:
        seen.add(id(origin))
        origin = origin.__cause__
    if code is None:
        if isinstance(exc, AgentExecutionError):
            code = exc.code
        elif isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
            code = "interrupted"
        elif isinstance(exc, SystemExit):
            code = "process_exit_attempt"
        elif isinstance(origin, httpx.TimeoutException):
            code = "model_timeout"
        elif isinstance(origin, httpx.HTTPStatusError):
            code = "model_http_error"
        elif isinstance(origin, httpx.RequestError):
            code = "model_unreachable"
        else:
            code = "internal_agent_error"
    if code not in MESSAGES:
        code = "internal_agent_error"
    error = {"code": code, "type": type(origin).__name__, "message": MESSAGES[code]}
    if isinstance(origin, json.JSONDecodeError):
        # Persist parser location without response text, custom exception messages,
        # credentials or document content. Unknown messages use a fixed label.
        reason = {"Expecting ',' delimiter": "expected_comma",
                  "Expecting ':' delimiter": "expected_colon",
                  "Expecting value": "expected_value",
                  "Expecting property name enclosed in double quotes": "expected_property_name",
                  "Unterminated string starting at": "unterminated_string",
                  "Extra data": "extra_data"}.get(origin.msg, "invalid_json")
        error['json_error'] = {'reason': reason, 'line': origin.lineno, 'column': origin.colno}
    if isinstance(origin, ValidationError):
        error['validation_fields'] = [{'type':item['type'], 'location':list(item['loc'])}
                                      for item in origin.errors(include_input=False, include_url=False)[:12]]
    if isinstance(origin, httpx.HTTPStatusError):
        error["http_status"] = origin.response.status_code
    if origin.__traceback__:
        error["frames"] = [{"file": Path(frame.filename).name, "function": frame.name, "line": frame.lineno}
                           for frame in traceback.extract_tb(origin.__traceback__)[-5:]]
    return error


def lifecycle_event(step, status, *, case_id=None, **fields):
    return {"step": step, "status": status, "case_id": case_id,
            "event_id": str(uuid4()), "timestamp": datetime.now(timezone.utc).isoformat(), **fields}


@contextmanager
def execution_audit(case_id, sink):
    events = []
    token = _context.set((case_id, sink, events))
    try:
        yield events
    finally:
        _context.reset(token)


def emit(status, *, step="agent_execution", **fields):
    context = _context.get()
    if context is None:
        return None
    case_id, sink, events = context
    event = lifecycle_event(step, status, case_id=case_id, **{**_agent.get(), **fields})
    if sink:
        sink(event)
    events.append(event)
    logger.info("Execution event; case=%s agent=%s status=%s event_id=%s error_code=%s",
                case_id, event.get("agent_key"), status, event["event_id"], (event.get("error") or {}).get("code"))
    return event


@contextmanager
def agent_execution(spec, settings, mode=None):
    token = _agent.set({"agent_key": spec.key, "agent": spec.name,
                         "execution_id": str(uuid4()), "provider": settings.llm_provider,
                         "model": settings.model_name, "detail": {"execution_mode": mode or "agent_task"}})
    started = perf_counter()
    try:
        emit("agent_started", duration_ms=0)
        yield
    except BaseException as exc:
        emit("execution_interrupted" if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)) else "agent_failed",
             duration_ms=round((perf_counter() - started) * 1000, 3), error=safe_error(exc))
        raise
    else:
        emit("agent_completed", duration_ms=round((perf_counter() - started) * 1000, 3))
    finally:
        _agent.reset(token)


@contextmanager
def model_request(*, attempt):
    """Surround one physical provider request, including each transient retry."""
    started = perf_counter()
    request_id = str(uuid4())
    result = {}
    emit("waiting_for_model_response", request_id=request_id, attempt=attempt, duration_ms=0)
    try:
        yield result
    except BaseException as exc:
        emit("model_request_interrupted" if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)) else "model_request_failed",
             request_id=request_id, attempt=attempt,
             duration_ms=round((perf_counter() - started) * 1000, 3), error=safe_error(exc))
        raise
    else:
        status = result.get("http_status", 200)
        error = ({"code": "model_http_error", "type": "HTTPStatusError", "http_status": status,
                  "message": MESSAGES["model_http_error"]} if not 200 <= status < 300 else None)
        emit("model_request_failed" if error else "model_response_received", request_id=request_id, attempt=attempt,
             duration_ms=round((perf_counter() - started) * 1000, 3), error=error)
