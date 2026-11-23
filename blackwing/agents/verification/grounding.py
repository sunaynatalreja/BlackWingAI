"""Final deterministic cross-check for material claims and human review."""
from __future__ import annotations

from typing import Any, Mapping

from blackwing.agents.verification.schema import VerificationOutput


def ground_verification_output(output: Mapping[str, Any], prior: Mapping[str, Any],
                              narrative: str = "") -> dict[str, Any]:
    result = VerificationOutput.model_validate(output).model_dump(mode="json")
    needs = list(result["claims_needing_source"])
    unsupported = list(result["unsupported_conclusions"])
    conflicts = list(result["conflicts"])
    # SIM-E labels and the synthetic case ID are source labels, not competing
    # real-world incident identities. Do not carry that common small-model
    # misread into the report as a factual conflict.
    from blackwing.case_inputs import is_synthetic_case
    if is_synthetic_case(narrative):
        conflicts = [text for text in conflicts
                     if not ("identit" in text.casefold() and "conflict" in text.casefold())]
    inv = prior.get("investigation") or {}
    for field in ("documented_facts", "reported_observations", "contributing_factors"):
        for claim in inv.get(field, []):
            if not claim.get("evidence_ids"):
                needs.append(str(claim.get("text") or "Uncited investigation statement"))
    for item in inv.get("timeline_events", []):
        if not item.get("evidence_ids"):
            needs.append("Unverified timeline: " + str(item.get("event") or ""))
    for hazard in (prior.get("risk") or {}).get("hazards", []):
        if not hazard.get("evidence_ids"):
            unsupported.append("Unverified potential hazard: " + str(hazard.get("hazard_id") or ""))
    for rec in (prior.get("mitigation") or {}).get("recommendations", []):
        if not rec.get("hazard_ids") and not rec.get("contributing_factor"):
            unsupported.append("Unlinked proposed mitigation: " + str(rec.get("action") or ""))
    for issue in (prior.get("critic") or {}).get("issues", []):
        if issue.get("issue_type") == "ungrounded_fact":
            conflicts.append("Investigation includes at least one ungrounded factual claim.")
    comp = prior.get("compliance") or {}
    if comp.get("overall_applicability") in ("not_assessed", None):
        needs.append("Applicable regulatory/procedural source and date require verification.")
    result["claims_needing_source"] = list(dict.fromkeys(needs))[:20]
    result["unsupported_conclusions"] = list(dict.fromkeys(unsupported))[:20]
    result["conflicts"] = list(dict.fromkeys(conflicts))[:20]
    # Structural checks cannot certify truth or approve consequential findings.
    result["completeness_status"] = "incomplete" if result["claims_needing_source"] or result["unsupported_conclusions"] else "pending_human_verification"
    result["human_approval_required"] = True
    return VerificationOutput.model_validate(result).model_dump(mode="json")
