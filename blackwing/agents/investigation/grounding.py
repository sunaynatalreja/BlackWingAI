"""Deterministic investigation evidence sanitization; no second LLM needed."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from blackwing.agents.evidence_utils import is_direct, resolve
from blackwing.agents.investigation.schema import InvestigationOutput


def ground_investigation_output(output: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]],
                                *, incident_id: str | None = None) -> dict[str, Any]:
    result = InvestigationOutput.model_validate(output).model_dump(mode="json")
    has_current_factual = any(is_direct(row, incident_id) for row in evidence)
    unknowns = list(result["unknowns"])
    for field in ("documented_facts", "reported_observations", "hypotheses", "contributing_factors"):
        sanitized = []
        for claim in result[field]:
            old = list(claim.get("evidence_ids") or [])
            ids, sources, rejected = resolve(old, evidence, direct_only=True, incident_id=incident_id)
            claim["evidence_ids"] = ids
            # No naked source IDs: derived only from the citations retained.
            claim["source_ids"] = sources
            if field == "hypotheses":
                claim["evidence_state"] = "hypothesis"
            elif field == "contributing_factors" and not ids:
                claim["evidence_state"] = "hypothesis"
            elif field in {"documented_facts", "reported_observations"}:
                if not ids:
                    # Preserve the statement as an unverified lead, never a fact.
                    claim["evidence_state"] = "unknown"
                elif field == "reported_observations":
                    claim["evidence_state"] = "reported_observation"
                elif any(str(row.get("source_type")) in {"user_submitted", "user_report"}
                         for row in evidence if str(row.get("chunk_id") or row.get("evidence_id")) in ids):
                    claim["evidence_state"] = "reported_observation"
                else:
                    claim["evidence_state"] = "documented_fact"
            if rejected:
                note = "Invalid or non-current-case citations excluded; human source review required."
                claim["rationale"] = (str(claim.get("rationale") or "").strip() + " " + note).strip()
            if not ids and field in {"documented_facts", "reported_observations"}:
                unknowns.append(f"Unverified statement: {claim['text']}")
            sanitized.append(claim)
        result[field] = sanitized
    for event in result["timeline_events"]:
        ids, _, rejected = resolve(event.get("evidence_ids") or [], evidence,
                                   direct_only=True, incident_id=incident_id)
        event["evidence_ids"] = ids
        if not ids:
            event["evidence_state"] = "unknown"
            unknowns.append("Timeline detail requires primary-source verification: " + event["event"])
        elif event["evidence_state"] == "documented_fact":
            # An exact citation alone does not prove the timestamp; reviewer verifies.
            event["evidence_state"] = "reported_observation"
    if not has_current_factual:
        unknowns.append("No authenticated current-incident factual passage was supplied to Investigation.")
    result["unknowns"] = list(dict.fromkeys(unknowns))[:15]
    return InvestigationOutput.model_validate(result).model_dump(mode="json")
