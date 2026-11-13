"""Connect saved case findings without inferring risk or missing evidence."""


def build_hazard_register(findings):
    """Return a hazard-centred view of the case's saved findings."""
    linked = {}
    for finding in findings:
        for hazard_id in finding.get("related_finding_ids") or ():
            linked.setdefault(hazard_id, []).append(finding)

    rows = []
    for hazard in findings:
        if hazard.get("kind") not in {"hazard", "potential_hazard"}:
            continue
        children = linked.get(hazard["finding_id"], [])
        status = hazard.get("status", "pending")
        if status == "accepted":
            verification = "Accepted by reviewer"
        elif status == "rejected":
            verification = "Rejected by reviewer"
        elif hazard.get("verification", {}).get("eligible") is False:
            verification = "Evidence check flagged"
        else:
            verification = "Pending human review"

        rows.append({
            "hazard": hazard,
            "supporting_evidence": hazard.get("citations") or [],
            "potential_consequence": hazard.get("potential_consequence") or "",
            "existing_controls": hazard.get("existing_controls") or [],
            "risk_assessment": [item for item in children if item.get("kind") == "risk"],
            "proposed_mitigations": [item for item in children if item.get("section") == "mitigation"],
            "uncertainty": hazard.get("uncertainty") or "Uncertainty has not been documented.",
            "verification_status": verification,
        })
    return rows
