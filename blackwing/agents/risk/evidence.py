"""Deterministic RAG grounding guard for preliminary risk findings.

No inference API, embedding call, or risk matrix is needed here. This guards
citation integrity after the existing Risk Agent returns its typed JSON.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from blackwing.agents.risk.schema import RiskOutput

_DIRECT_TYPES = {"official_factual", "official_preliminary", "user_submitted", "user_report"}
_ADVISORY_TYPES = {"advisory", "guidance"}
_HISTORICAL_ROLES = {"historical_analogy", "similar_case", "historical"}
_HISTORICAL_TYPES = {"official_final", "official_preliminary", "official_factual", "voluntary_asrs"}
_INCIDENT_DOCUMENT_TYPES = {"incident_report", "asrs_report", "incident_summary", "safety_report"}


def _reference(item: Mapping[str, Any]) -> str:
    """Use the exact passage ID given to the model, never invent a new ID."""
    return str(item.get("chunk_id") or item.get("evidence_id") or "").strip()


def _type(item: Mapping[str, Any]) -> str:
    return str(item.get("source_type") or item.get("document_type") or "").casefold()


def _is_usable(item: Mapping[str, Any]) -> bool:
    return (str(item.get("evidence_role") or "").casefold() in _HISTORICAL_ROLES
            or _type(item) in _DIRECT_TYPES | _ADVISORY_TYPES | _HISTORICAL_TYPES)


def _is_historical(item: Mapping[str, Any]) -> bool:
    # A role label alone is not enough: guidance can accidentally receive the
    # historical_analogy label during retrieval. Only actual incident records
    # with an incident identity may support an analogy-based rating.
    kind = _type(item)
    doc_type = str(item.get("document_type") or "").casefold()
    incident_id = str(item.get("incident_id") or item.get("case_id") or "").strip()
    return (
        kind in _HISTORICAL_TYPES
        and doc_type in _INCIDENT_DOCUMENT_TYPES
        and bool(incident_id)
        and str(item.get("evidence_role") or "").casefold() in _HISTORICAL_ROLES
        and str(item.get("evidence_status") or "").casefold() not in {"gold", "evaluation_only", "protected_gold"}
    )


def historical_case_key(item: Mapping[str, Any]) -> str | None:
    """Return a distinct, eligible incident key for analogue counting."""
    if not _is_historical(item):
        return None
    return str(item.get("incident_id") or item.get("case_id") or "").strip() or None


def ground_risk_output(
    output: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Prevent cross-case or made-up citations in a Risk Agent result.

    A hazard can remain a *potential* hazard based on the user's narrative,
    but an unsupported hazard is not converted into an observed factual event.
    Existing controls and control gaps remain provisional textual suggestions:
    no deterministic claim-level semantic verification is asserted.
    """
    result = RiskOutput.model_validate(output)
    allowed = {_reference(item): item for item in evidence if _reference(item) and _is_usable(item)}
    source_aliases: dict[str, list[str]] = {}
    for item in evidence:
        ref = _reference(item)
        source = str(item.get("source_id") or "").strip()
        if ref in allowed and source and source != ref:
            source_aliases.setdefault(source, []).append(ref)
    note: list[str] = []
    missing = list(result.missing_inputs)
    any_direct_citation = False
    for hazard in result.hazards:
        provided = list(hazard.evidence_ids)
        hazard.evidence_ids = list(dict.fromkeys(
            ref for ref in provided
            if ref in allowed and _type(allowed[ref]) in _DIRECT_TYPES and not _is_historical(allowed[ref])
        ))
        if len(hazard.evidence_ids) < len(set(provided)):
            note.append("Historical analogy IDs are not current-incident evidence; use rating_evidence_ids for contextual ratings.")
        rating_provided = list(hazard.rating_evidence_ids)
        canonical_rating_refs = []
        invalid_rating_refs = []
        for ref in rating_provided:
            if ref in allowed and (_is_historical(allowed[ref]) or _type(allowed[ref]) in _DIRECT_TYPES):
                canonical_rating_refs.append(ref)
            elif ref in source_aliases:
                canonical_rating_refs.extend(
                    alias for alias in source_aliases[ref]
                    if _is_historical(allowed[alias]) or _type(allowed[alias]) in _DIRECT_TYPES
                )
            else:
                invalid_rating_refs.append(ref)
        hazard.rating_evidence_ids = list(dict.fromkeys(canonical_rating_refs))
        if invalid_rating_refs:
            note.append("Removed rating references that were not present in the supplied evidence packet.")
        direct = any(_type(allowed[ref]) in _DIRECT_TYPES for ref in hazard.evidence_ids)
        any_direct_citation = any_direct_citation or direct
        if not direct:
            missing.append(
                f"Confirm incident-specific factual evidence for potential hazard {hazard.hazard_id}."
            )
    if result.hazards and not any_direct_citation:
        note.append("Candidate hazards are hypotheses, not established current-incident findings.")
    # Deterministic risk assessment after citation validation.
    from blackwing.tools import assess_reference_risk

    classified = 0
    available_historical_refs = [ref for ref, item in allowed.items() if _is_historical(item)]
    available_historical_cases = {str(allowed[ref].get("incident_id") or allowed[ref].get("source_id") or ref)
                                 for ref in available_historical_refs}

    for hazard in result.hazards:
        # Historical analogies and guidance cannot establish
        # incident-specific severity or likelihood.
        direct_ids = [
            ref for ref in hazard.evidence_ids
            if _type(allowed[ref]) in _DIRECT_TYPES
            and not ref.startswith("CASE-NARRATIVE:")
            and allowed[ref].get("evidence_status") != "SYNTHETIC_UNVERIFIED"
        ]
        historical_refs = [ref for ref in hazard.rating_evidence_ids if _is_historical(allowed[ref])]
        historical_cases = {str(allowed[ref].get("incident_id") or allowed[ref].get("source_id") or ref)
                            for ref in historical_refs}
        rating_ids = direct_ids
        if not rating_ids and len(historical_cases) >= 2:
            rating_ids = historical_refs
            hazard.rating_basis_type = "historical_analogies"
            note.append("Provisional category is analogy-informed; historical cases do not establish this occurrence or a statistical frequency.")
        elif rating_ids:
            hazard.rating_basis_type = "direct_case_evidence"
        else:
            hazard.rating_basis_type = "unassessed"

        assessment = assess_reference_risk(
            severity=hazard.severity,
            likelihood=hazard.likelihood,
            evidence_ids=rating_ids,
            severity_basis=hazard.severity_basis,
            likelihood_basis=hazard.likelihood_basis,
        )

        hazard.risk_index = assessment["risk_index"]
        hazard.risk_level = assessment["risk_level"]
        hazard.scoring_status = assessment["status"]

        if hazard.risk_level is not None:
            classified += 1
            hazard.scoring_reason = (
                "Provisional matrix category supported by cited historical analogues; "
                "this is not a statistical frequency or a finding about the current incident."
                if hazard.rating_basis_type == "historical_analogies" else
                "Provisional matrix category supported by current-case evidence; human review is required."
            )
        else:
            # A synthetic account alone cannot support a categorical rating;
            # an analogy-based rating requires two distinct historical cases.
            if len(available_historical_cases) < 2:
                hazard.scoring_reason = (
                    f"Only {len(available_historical_cases)} distinct historical incident source(s) were available; "
                    "the provisional analogy route requires at least two."
                )
            elif hazard.severity is None or hazard.likelihood is None:
                hazard.scoring_reason = (
                    "At least two historical sources were available, but the Risk Agent returned no complete "
                    "severity/likelihood pair and cited rating basis. The backend withheld the index."
                )
            elif len(historical_cases) < 2 and not rating_ids:
                hazard.scoring_reason = (
                    "At least two historical sources were available, but the Risk Agent did not cite two "
                    "distinct rating passages. The backend withheld the index."
                )
            elif not rating_ids:
                hazard.scoring_reason = (
                    "Ratings were returned without sufficient validated current-case or analogy citations; "
                    "the backend withheld the index."
                )
            else:
                hazard.scoring_reason = "A source-based severity or likelihood rationale was missing; the backend withheld the index."
            hazard.severity = None
            hazard.likelihood = None
            hazard.severity_basis = ""
            hazard.likelihood_basis = ""

    # FAA reference classification is categorical, not numeric.
    result.numeric_score = None
    result.assessment_status = (
        "provisional_reference_matrix_requires_human_review"
        if classified
        else "unassessed_insufficient_inputs"
    )
    if note:
        result.risk_notes = (result.risk_notes + " " + " ".join(dict.fromkeys(note))).strip()
    result.missing_inputs = list(dict.fromkeys(missing))[:12]
    return result.model_dump(mode="json")
