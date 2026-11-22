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


def _reference(item: Mapping[str, Any]) -> str:
    """Use the exact passage ID given to the model, never invent a new ID."""
    return str(item.get("chunk_id") or item.get("evidence_id") or "").strip()


def _type(item: Mapping[str, Any]) -> str:
    return str(item.get("source_type") or item.get("document_type") or "").casefold()


def _is_usable(item: Mapping[str, Any]) -> bool:
    if str(item.get("evidence_role") or "").casefold() in _HISTORICAL_ROLES:
        return False
    return _type(item) in _DIRECT_TYPES | _ADVISORY_TYPES


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
    note: list[str] = []
    missing = list(result.missing_inputs)
    any_direct_citation = False
    for hazard in result.hazards:
        provided = list(hazard.evidence_ids)
        hazard.evidence_ids = list(dict.fromkeys(
            ref for ref in provided if ref in allowed
        ))
        if len(hazard.evidence_ids) < len(set(provided)):
            note.append("Removed unsupported or cross-incident hazard evidence references.")
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

    for hazard in result.hazards:
        # Historical analogies and guidance cannot establish
        # incident-specific severity or likelihood.
        direct_ids = [
            ref for ref in hazard.evidence_ids
            if _type(allowed[ref]) in _DIRECT_TYPES
            and not ref.startswith("CASE-NARRATIVE:")
            and allowed[ref].get("evidence_status") != "SYNTHETIC_UNVERIFIED"
        ]

        assessment = assess_reference_risk(
            severity=hazard.severity,
            likelihood=hazard.likelihood,
            evidence_ids=direct_ids,
            severity_basis=hazard.severity_basis,
            likelihood_basis=hazard.likelihood_basis,
        )

        hazard.risk_index = assessment["risk_index"]
        hazard.risk_level = assessment["risk_level"]
        hazard.scoring_status = assessment["status"]

        if hazard.risk_level is not None:
            classified += 1
        else:
            # Do not publish unsupported model-generated ratings.
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
