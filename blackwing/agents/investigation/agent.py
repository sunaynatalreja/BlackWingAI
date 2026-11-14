"""Post-process investigation citations to separate historical analogies from case evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from blackwing.agents.base import AgentSpec
from blackwing.agents.investigation.schema import InvestigationOutput


SPEC = AgentSpec(
    key="investigation",
    name="Investigation Agent",
    prompt_path=Path(__file__).with_name("prompt.md"),
    output_schema=InvestigationOutput,
    upstream_context=(),
)


def remove_historical_case_evidence_ids(
    output: dict[str, Any], evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    """A retrieved historical report may inspire a hypothesis, never prove a new fact."""
    historical_sources: dict[str, str] = {}
    for item in evidence:
        if item.get("chunk_id"):
            source_id = str(item.get("source_id") or item.get("document_id") or "historical report")
            aliases = [item.get("chunk_id"), item.get("document_id")]
            # Some small models cite the zero-based chunk_index instead of the
            # public one-based chunk_id suffix; map that common variant too.
            if item.get("document_id") is not None and item.get("chunk_index") is not None:
                aliases.append(f"{item['document_id']}:{item['chunk_index']}")
            for identifier in aliases:
                if identifier:
                    historical_sources[str(identifier)] = source_id
    if not historical_sources:
        return output

    sanitized = {**output}
    for key in (
        "documented_facts", "reported_observations", "hypotheses", "contributing_factors"
    ):
        claims = []
        for original in output.get(key, []):
            claim = dict(original)
            cited = [str(identifier) for identifier in claim.get("evidence_ids", [])]
            historical = [
                identifier for identifier in cited
                if identifier in historical_sources
                or any(identifier.startswith(f"{item.get('document_id')}:") for item in evidence if item.get("chunk_id"))
            ]
            if historical:
                claim["evidence_ids"] = [identifier for identifier in cited if identifier not in historical]
                source_ids = list(claim.get("source_ids", []))
                for identifier in historical:
                    source_ids.append(historical_sources.get(identifier) or next(
                        str(item.get("source_id") or item.get("document_id") or "historical report")
                        for item in evidence
                        if item.get("chunk_id") and identifier.startswith(f"{item.get('document_id')}:")
                    ))
                claim["source_ids"] = sorted(set(source_ids))
                if key in {"hypotheses", "contributing_factors"}:
                    claim["evidence_state"] = "hypothesis"
                    note = "Historical analogy only; this source describes a different incident and does not establish this factor in the current case. Verify using current-case records."
                else:
                    claim["evidence_state"] = "unknown"
                    note = "Historical report citation removed as direct support: it describes a different incident, not the current case."
                prior = str(claim.get("rationale") or "").strip()
                claim["rationale"] = f"{prior} {note}".strip()
            claims.append(claim)
        sanitized[key] = claims

    timeline = []
    for original in output.get("timeline_events", []):
        event = dict(original)
        cited = [str(identifier) for identifier in event.get("evidence_ids", [])]
        historical = [
            identifier for identifier in cited
            if identifier in historical_sources
            or any(identifier.startswith(f"{item.get('document_id')}:") for item in evidence if item.get("chunk_id"))
        ]
        if historical:
            event["evidence_ids"] = [identifier for identifier in cited if identifier not in historical]
            event["evidence_state"] = "unknown"
            event["event"] = f"{event.get('event', '')} (historical-report support removed; verify against current-case records)".strip()
        timeline.append(event)
    sanitized["timeline_events"] = timeline
    return sanitized
