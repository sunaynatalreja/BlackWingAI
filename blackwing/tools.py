"""Executable tools with invocation-scoped, credential-free audit records."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

_context = ContextVar("tool_audit", default=(None, None))


@contextmanager
def tool_audit(case_id, sink):
    token = _context.set((case_id, sink))
    try:
        yield
    finally:
        _context.reset(token)


def identifiers(rows):
    return [str(row.get("chunk_id") or row.get("evidence_id") or
                row.get("document_id") or row.get("hazard_id") or "unknown") for row in rows]


def execute_tool(name, function, *, input_identifiers=(), **kwargs):
    case_id, sink = _context.get()
    event = {"step": name, "tool": name, "invocation_id": str(uuid4()),
             "case_id": case_id, "input_identifiers": list(input_identifiers),
             "timestamp": datetime.now(timezone.utc).isoformat()}
    started = perf_counter()
    try:
        result = function(**kwargs)
        event.update(status="complete", result_summary=_summary(result), error=None)
        return result
    except Exception as exc:
        # Exception messages and provider response bodies can contain credentials.
        event.update(status="failed", result_summary=None, error={"type": type(exc).__name__})
        raise
    finally:
        event["duration_ms"] = round((perf_counter() - started) * 1000, 3)
        if sink is not None:
            sink(event)


def _summary(result):
    if isinstance(result, dict):
        if "risk_index" in result:
            return {key: result.get(key) for key in ("status", "risk_index", "risk_level")}
        if "results" in result:
            return {"output_identifiers": identifiers(result["results"]), "count": len(result["results"])}
        return {"fields": list(result), "item_counts": {
            key: len(value) for key, value in result.items() if isinstance(value, list)}}
    return {"type": type(result).__name__}


def retrieve_evidence(function, *, input_identifiers=(), **kwargs):
    return execute_tool("retrieve_evidence", function, input_identifiers=input_identifiers, **kwargs)


def find_similar_incidents(*, narrative, evidence, current_incident_id):
    from blackwing.agents.similar_case.agent import build_output
    return execute_tool("find_similar_incidents", build_output,
                        input_identifiers=identifiers(evidence), narrative=narrative,
                        evidence=evidence, current_incident_id=current_incident_id)


def assess_reference_risk(**kwargs):
    from blackwing.agents.risk.matrix import assess_reference_risk as calculate
    return execute_tool("assess_reference_risk", calculate,
                        input_identifiers=kwargs.get("evidence_ids") or [], **kwargs)
