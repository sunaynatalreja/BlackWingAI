"""Deterministic applicability and citation checks for Compliance Agent output.

Only source passages already supplied by RAG can support a compliance check.
Neither rules nor guidance are declared legally applicable by this validator.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence

from blackwing.agents.compliance.schema import ComplianceOutput

_RULE_TYPES = {"regulation", "operational_order", "procedure"}
_ADVISORY_TYPES = {"advisory", "guidance"}
_HISTORICAL_ROLES = {"historical_analogy", "similar_case", "historical"}


def _kind(item: Mapping[str, Any]) -> str:
    return str(item.get("source_type") or item.get("document_type") or "").strip().casefold()


def _date(s: Any) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


def _passage_id(item: Mapping[str, Any]) -> str:
    return str(item.get("chunk_id") or item.get("evidence_id") or "").strip()


def _valid_source(item: Mapping[str, Any], *, jurisdiction: str | None, incident_date: str | None) -> bool:
    role = str(item.get("evidence_role") or "").casefold()
    if role in _HISTORICAL_ROLES:
        return False
    kind = _kind(item)
    when = _date(incident_date)
    publication = _date(item.get("publication_date") or item.get("document_date"))
    if kind in _RULE_TYPES:
        # A binding source cannot be assessed without its governing context.
        if not jurisdiction or not when:
            return False
        if str(item.get("jurisdiction") or "").casefold() != jurisdiction.casefold():
            return False
        effective = _date(item.get("effective_date") or publication)
        if effective is None or effective > when:
            return False
        end = _date(item.get("effective_through"))
        return end is None or end >= when
    if kind in _ADVISORY_TYPES:
        # General guidance is context only. A future publication may not be
        # used as contemporaneous guidance for an older incident.
        return not (when and (publication is None or publication > when))
    return False


def ground_compliance_output(
    output: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    *,
    jurisdiction: str | None,
    incident_date: str | None,
) -> dict[str, Any]:
    """Remove non-existent/out-of-date sources, bound uncertain compliance claims.

    This is mechanical source checking, NOT legal analysis or textual NLI.
    Incident narratives/user reports are not themselves regulatory sources.
    """
    result = ComplianceOutput.model_validate(output)
    sources = {
        str(item.get("source_id") or "").strip(): item
        for item in evidence if item.get("source_id") and _valid_source(
            item, jurisdiction=jurisdiction, incident_date=incident_date
        )
    }
    # Exact references to supplied incident evidence or legitimate rule passages.
    known_passages = {_passage_id(item) for item in evidence if _passage_id(item)}
    checks = []
    missing = list(result.missing_context)
    for check in result.checks:
        item = sources.get(str(check.source_id or ""))
        if item is None:
            missing.append("A proposed compliance source was not supplied or is not date/jurisdiction eligible.")
            continue
        kind = _kind(item)
        check.source_type = kind
        check.source_title = str(item.get("title") or item.get("original_filename") or check.source_title or "") or None
        check.jurisdiction = str(item.get("jurisdiction") or "") or None
        check.effective_date = str(item.get("effective_date") or "") or None
        check.evidence_ids = list(dict.fromkeys(ref for ref in check.evidence_ids if ref in known_passages))
        passage = _passage_id(item)
        if passage and passage not in check.evidence_ids:
            check.evidence_ids.insert(0, passage)
        if kind in _ADVISORY_TYPES:
            check.applicability_status = "advisory_only"
            check.rationale = (check.rationale + " Advisory text is not a binding local requirement.").strip()
        else:
            check.applicability_status = "requires_human_verification"
            # A quoted clause must actually appear in the supplied source.
            section = (check.section or "").strip()
            source_section = str(item.get("section_reference") or "")
            source_text = str(item.get("excerpt") or item.get("content") or "")
            if section and section.casefold() not in (source_section + " " + source_text).casefold():
                check.section = None
                missing.append("Confirm the exact regulation/procedure section from the original source.")
        checks.append(check)
    result.checks = checks
    result.jurisdiction = jurisdiction
    result.overall_applicability = (
        "requires_human_verification" if any(_kind(sources[c.source_id]) in _RULE_TYPES for c in checks)
        else "advisory_only" if checks else "not_assessed"
    )
    if not checks:
        missing.append("No eligible, date-aware regulatory or advisory source was confirmed.")
    if not jurisdiction or not incident_date:
        missing.append("Confirm incident jurisdiction and date before interpreting regulations.")
    result.missing_context = list(dict.fromkeys(missing))[:12]
    return result.model_dump(mode="json")
