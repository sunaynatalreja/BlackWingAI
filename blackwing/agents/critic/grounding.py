"""Deterministic cross-agent audit; augments, never whitewashes model Critic."""
from __future__ import annotations

from typing import Any, Mapping
from blackwing.agents.critic.schema import CriticOutput


def audit_critic_output(output: Mapping[str, Any], prior: Mapping[str, Any]) -> dict[str, Any]:
    result = CriticOutput.model_validate(output).model_dump(mode="json")
    issues = list(result["issues"])
    unique = {(i["issue_type"], i.get("affected_output")) for i in issues}

    def add(kind: str, role: str, description: str, correction: str) -> None:
        if (kind, role) not in unique:
            issues.append({"issue_type": kind, "severity": "review", "affected_output": role,
                           "description": description, "suggested_correction": correction})
            unique.add((kind, role))

    inv = prior.get("investigation") or {}
    facts = list(inv.get("documented_facts", []))
    if not any(f.get("evidence_ids") and f.get("evidence_state") == "documented_fact" for f in facts):
        add("missing_current_case_evidence", "investigation", "No current-incident documented fact has a valid citation.",
            "Supply a primary event record and re-run Investigation.")
    for claim in facts:
        if not claim.get("evidence_ids") or claim.get("evidence_state") != "documented_fact":
            add("ungrounded_fact", "investigation", "At least one proposed factual claim lacks direct-case evidence.",
                "Demote to unknown or attach an actual passage ID.")
            break

    risk = prior.get("risk") or {}
    if any(not h.get("evidence_ids") for h in risk.get("hazards", [])):
        add("ungrounded_hazard", "risk", "One or more candidate hazards have no supporting incident evidence ID.",
            "Retain as hypotheses and collect incident-specific evidence.")
    comp = prior.get("compliance") or {}
    if comp.get("overall_applicability") in (None, "not_assessed"):
        add("regulatory_context_gap", "compliance", "Regulatory applicability was not assessed.",
            "Retrieve source versions and verify jurisdiction and incident date; do not allege violations.")
    mitigation = prior.get("mitigation") or {}
    for rec in mitigation.get("recommendations", []):
        if not rec.get("hazard_ids") and not rec.get("contributing_factor"):
            add("unlinked_mitigation", "mitigation", "Proposed action is not linked to a hazard or factor.",
                "Remove or explicitly link to a documented safety concern.")
            break
    if not mitigation.get("recommendations"):
        add("no_grounded_mitigations", "mitigation", "No evidence-linked mitigation option is ready for review.",
            "Collect hazard/barrier information and propose traceable options.")

    result["issues"] = issues[:20]
    result["overall_assessment"] = "requires_human_review"
    if result["issues"] and "Investigate audit issues before treating conclusions as findings." not in result["missing_analysis"]:
        result["missing_analysis"].append("Investigate audit issues before treating conclusions as findings.")
    return CriticOutput.model_validate(result).model_dump(mode="json")
