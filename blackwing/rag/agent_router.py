from __future__ import annotations

from typing import Any, Mapping

AGENT_ROLES = (
    "investigation", "similar_case", "compliance", "risk", "mitigation",
    "critic", "verification",
)

# Query wording is generic and applies to any runway-incursion incident.
TOPIC_QUERIES = {
    "direct": "runway taxiway aircraft vehicle ATC clearance readback event chronology surface movement factual evidence",
    "historical": "historical runway incursion runway occupancy conflict aircraft vehicle takeoff landing taxi route clearance",
    "regulatory": "runway entry clearance hold short readback airport movement area air traffic control requirements procedures",
    "guidance": "runway incursion hazard safety barriers situational awareness signage surveillance communication preventive measures",
}


def _unique(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = str(item.get("chunk_id") or item.get("evidence_id") or item.get("document_id") or item.get("source_id") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        results.append(item)
        if len(results) == limit:
            break
    return results


def _case_uploads(items: list[dict[str, Any]], case_id: str) -> list[dict[str, Any]]:
    """User attachments are reported observations, not official factual findings."""
    uploads = []
    for row in items:
        if not row.get("evidence_id") or not row.get("excerpt"):
            continue
        uploads.append({
            **row,
            "source_id": row.get("source_id") or row["evidence_id"],
            "source_type": "user_submitted",
            "evidence_role": "reported",
            "incident_id": case_id,
            "evidence_status": "USER_REPORTED_UNVERIFIED",
        })
    return uploads


def _safe_query(narrative: str, topic: str) -> str:
    # Use identifiers from narrative when available, but avoid sending full
    # lengthy evidence or sensitive input to an external embeddings endpoint.
    head = " ".join(narrative.split())[:350]
    return f"{topic} {head}"


def build_agent_packets(*, retriever: Any, narrative: str, case_id: str,
                        jurisdiction: str | None, incident_date: str | None,
                        documents: list[dict[str, Any]],
                        uploaded_evidence: list[dict[str, Any]] | None = None,
                        embedding_model: str = "BAAI/bge-small-en-v1.5",
                        limit_per_agent: int = 3,
                        method: str = "hybrid") -> dict[str, list[dict[str, Any]]]:
    """Route narrowly retrieved evidence to ALL seven existing agents.

    Retrieval stays in Python; the small Qwen model only sees the 1--3 passages
    relevant to its role. Backend can be local or IIT, as explicitly configured
    in the existing retriever. Never silently treat historical as current evidence.
    """
    if not 1 <= limit_per_agent <= 5:
        raise ValueError("limit_per_agent must be 1..5 to respect small Qwen context")
    if not case_id:
        raise ValueError("Case identity is required for source separation")
    uploads = _case_uploads(uploaded_evidence or [], case_id)

    def search(scope: str, limit: int, **kwargs: Any) -> list[dict[str, Any]]:
        result = retriever.search(
            _safe_query(narrative, TOPIC_QUERIES[scope]), documents,
            embedding_model=embedding_model, limit=limit, method=method,
            scope=scope, **kwargs,
        )
        return [dict(row) for row in result["results"]]

    direct = search("direct", 2, case_id=case_id, jurisdiction=jurisdiction,
                    incident_date=incident_date)
    historical = search("historical", 2, case_id=case_id,
                        incident_date=incident_date)
    for row in historical:
        row["evidence_role"] = "historical_analogy"
        row["caution"] = "OTHER INCIDENT: analogy only; never factual evidence for current case"
    rules = search("regulatory", 2, jurisdiction=jurisdiction,
                   incident_date=incident_date) if jurisdiction and incident_date else []
    guidance = search("guidance", 2, incident_date=incident_date)
    for row in guidance:
        row["evidence_role"] = "general_guidance"
        row["caution"] = "Advisory, not a binding determination"
    current = _unique([*uploads, *direct], limit_per_agent)

    # One coherent, bounded evidence packet per agent; similar_case is
    # deterministic in the existing service. It must never receive case uploads.
    return {
        "investigation": current,
        "similar_case": _unique(historical, limit_per_agent),
        # A compliance comparison needs both the rule and the current incident
        # evidence; guidance is supplementary and never silently binding.
        "compliance": _unique([*rules[:2], *current[:1], *guidance], limit_per_agent),
        "risk": _unique([*current[:2], *guidance], limit_per_agent),
        "mitigation": _unique([*guidance, *current[:1]], limit_per_agent),
        "critic": _unique([*current[:2], *rules], limit_per_agent),
        "verification": _unique([*current[:2], *rules], limit_per_agent),
    }


def packet_diagnostics(packets: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Structural check only; NOT retrieval Recall or model report accuracy."""
    return {
        "all_seven_roles_present": set(packets) == set(AGENT_ROLES),
        "source_counts": {role: len(packets.get(role, [])) for role in AGENT_ROLES},
        "sources": {role: [item.get("citation_label") or item.get("source_id")
                           for item in packets.get(role, [])] for role in AGENT_ROLES},
        "quality_claim": "NOT VERIFIED: evaluate with real BGE, CrossEncoder and complete gold relevance labels",
    }
