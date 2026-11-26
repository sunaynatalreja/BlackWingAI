"""BLACKWING AI completeness rubric v2.2; original v2 weights retained.

Integration note: Drop-in replacement for assess_quality(incident, report=None),
not a source-entailment or aviation risk score. No artificial minimum score.
"""
from __future__ import annotations

from typing import Any


EXPECTED_AGENTS = frozenset({
    "investigation", "similar_case", "compliance", "risk",
    "mitigation", "critic", "verification",
})


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _has_claim_text(item: dict) -> bool:
    return any(isinstance(item.get(key), str) and item[key].strip()
               for key in ("text", "fact", "claim", "description"))


def _has_refs(item: dict) -> bool:
    # Check both supported ID fields; null/whitespace entries are not citations.
    return any(isinstance(ref, str) and ref.strip()
               for key in ("evidence_ids", "source_ids")
               if isinstance(item.get(key), (list, tuple, set))
               for ref in item[key])


def _safe_risk_record(hazard: dict) -> bool:
    """Reject unsupported ratings; permit documented abstention.

    Cannot prove whether citations support the severity and likelihood.
    """
    status = str(hazard.get("scoring_status") or "").lower()
    index = hazard.get("risk_index")
    level = hazard.get("risk_level")
    if status.startswith(("unassessed", "insufficient", "not_scored")) and not index and not level:
        return True
    if index or (level and str(level).lower() not in {"unassessed", "unknown", "not_scored"}):
        return bool(
            _has_refs(hazard)
            and hazard.get("severity_basis")
            and hazard.get("likelihood_basis")
        )
    return False


def assess_quality(incident: dict, report: dict | None = None) -> dict:
    """Transparent, independent structural-readiness scores (0..100).

    `report=None` checks prompt completeness, never a generated report.
    A report score is *not* a factual-accuracy or citation-entailment score.
    Attributed accounts earn documentation coverage, not verified-fact status.
    Empty entries and unknowns do not earn fact/account-coverage credit.
    At no point is the score forced to >=70.
    """
    incident = _dict(incident)
    checks: list[dict] = []

    def check(name: str, weight: int, met: bool, remedy: str, *, partial: int = 0) -> None:
        earned = weight if met else min(max(partial, 0), weight)
        checks.append({
            "check": name,
            "weight": weight,
            "earned": earned,
            "status": "present" if met else ("partial" if earned else "missing_or_needs_review"),
            "action": "" if met else remedy,
        })

    if report is None:
        check("Incident narrative", 25, len(str(incident.get("narrative") or "").strip()) >= 20,
              "Describe the reported event and distinguish known facts from unknowns.")
        check("Airport/location", 20, bool(incident.get("location") or incident.get("airport")),
              "Specify airport or location.")
        check("Incident date", 20, bool(incident.get("incident_date") or incident.get("event_date")),
              "Specify incident date if known; do not invent it.")
        check("Jurisdiction", 15, bool(incident.get("jurisdiction")),
              "Specify jurisdiction or flag as unknown for applicability review.")
        check("Source identification", 20, bool(incident.get("source_incident_id") or incident.get("evidence")),
              "Provide a source identifier or attach supporting records; identifiers alone are not proof.")
        label = "Prompt Completeness"
    else:
        report = _dict(report)
        outputs = _dict(report.get("agent_outputs"))
        inv = _dict(outputs.get("investigation"))
        compliance = _dict(outputs.get("compliance"))
        risk = _dict(outputs.get("risk"))
        mitigation = _dict(outputs.get("mitigation"))
        critic = _dict(outputs.get("critic"))
        verification = _dict(outputs.get("verification"))
        generated = _dict(report.get("generated_report"))
        sections = _list(generated.get("sections"))
        supplied_facts = [_dict(f) for f in _list(inv.get("documented_facts"))]
        facts = [f for f in supplied_facts if _has_claim_text(f)
                 and f.get("evidence_state", "documented_fact") == "documented_fact"]
        reported = [f for f in supplied_facts if _has_claim_text(f)
                    and f.get("evidence_state") == "reported_observation"]
        reported += [f for raw in _list(inv.get("reported_observations"))
                     if _has_claim_text(f := _dict(raw))
                     and f.get("evidence_state", "reported_observation") == "reported_observation"]
        case_evidence = [*facts, *reported]
        evidence_points = [
            _dict(p)
            for section in sections
            for p in _list(_dict(section).get("key_points"))
            if _dict(p).get("classification") in {"documented_fact", "reported_observation"}
            and _has_claim_text(_dict(p))
        ]
        # Completeness is evaluated on the published report. The investigation
        # agent's internal claim list is useful provenance, but a valid
        # attributed point may be copied into the report from supplied evidence
        # even when that agent conservatively marked its own claim unknown.
        # Source-entailment remains outside this structural score.
        published_case_evidence = evidence_points
        unknowns = _list(inv.get("unknowns"))
        approval_needed = verification.get("human_approval_required") is True

        check("Seven required agent outputs", 8, EXPECTED_AGENTS <= set(outputs),
              "Obtain missing agent outputs; report is a separately generated artifact.")
        check("Executive summary and report sections", 12,
              bool(generated.get("executive_summary")) and len(sections) >= 4,
              "Produce a summary and four required report sections.")
        check("Current-case evidence represented", 12,
              bool(case_evidence) or bool(published_case_evidence),
              "Document current-case facts or attributed accounts; disclose when neither is available.")
        check("Factual evidence references present", 13,
              bool(published_case_evidence) and all(_has_refs(point) for point in published_case_evidence),
              "Reference every published fact or attributed account; citation presence is not entailment.")
        check("Report factual points referenced", 10,
              bool(evidence_points) and all(_has_refs(p) for p in evidence_points),
              "Reference every factual or attributed report point; empty points cannot earn credit.")
        check("Human verification required", 8, approval_needed,
              "Keep explicit human approval required before final safety or regulatory conclusions.")
        check("Unknowns/completeness disclosed", 7,
              bool(unknowns) or verification.get("completeness_status") == "complete",
              "State missing or conflicting case evidence; do not invent answers.")

        # This verifies explicit workflow output, not actual jurisdictional legality.
        applicability_keys = (
            "applicability_status", "applicability", "uncertainties", "unknowns",
            "rules", "checks", "findings", "compliance_checks", "status",
            "overall_applicability", "missing_context",
        )
        check("Regulatory applicability review recorded", 8,
              any(bool(compliance.get(k)) for k in applicability_keys),
              "Show jurisdiction/version applicability or explicitly state it cannot be determined.")

        hazards = [_dict(h) for h in _list(risk.get("hazards"))]
        risk_status = str(risk.get("scoring_status") or risk.get("risk_scoring_status") or risk.get("assessment_status") or "").lower()
        no_rating = risk_status.startswith(("unassessed", "insufficient", "not_scored"))
        check("Risk rationale or explicit abstention", 8,
              (bool(hazards) and all(_safe_risk_record(h) for h in hazards)) or (not hazards and no_rating),
              "Record evidence-backed ratings or explicit unassessed status; do not manufacture a score.")

        hazard_ids = {h.get("hazard_id") for h in hazards if h.get("hazard_id")}
        recs = [_dict(r) for r in _list(mitigation.get("recommendations"))]
        all_linked = bool(recs) and all(
            (bool(r.get("hazard_ids")) and set(r["hazard_ids"]) <= hazard_ids)
            or bool(r.get("contributing_factor"))
            for r in recs
        )
        check("Mitigation traceability", 9, all_linked,
              "Link every recommendation to an existing hazard or evidenced factor; explicitly defer when data is insufficient.",
              partial=4 if not recs and bool(unknowns) else 0)

        issues = _list(critic.get("issues"))
        check("Critic review and issue handling", 5,
              bool(critic) and not issues,
              "Resolve critic issues or explicitly disclose them to reviewers; do not hide issues to gain points.",
              partial=2 if bool(critic) and bool(issues) and approval_needed else 0)
        label = "Report Completeness"

    assert sum(c["weight"] for c in checks) == 100
    score = sum(c["earned"] for c in checks)
    flags: list[str] = []
    if report is not None:
        if not published_case_evidence:
            score = min(score, 69)
            flags.append("No substantive cited current-case facts or attributed accounts are published: report cannot receive >=70.")
        elif not all(_has_refs(point) for point in published_case_evidence):
            score = min(score, 59)
            flags.append("Published current-case facts or attributed accounts have missing citations; critical integrity review required.")
        elif not facts and not any(
            _dict(point).get("classification") == "documented_fact" for point in published_case_evidence
        ):
            flags.append("Available case material is attributed reporting, not independently verified facts.")
    return {
        "label": label,
        "score": score,
        "raw_score": sum(c["earned"] for c in checks),
        "rubric_version": "2.2",
        "checks": checks,
        "missing_information": [c["action"] for c in checks if c["status"] != "present"],
        "flags": flags,
        "method": "Structural completeness v2.2; unchanged v2 weights; published attributed-account coverage is not verified accuracy.",
        "notice": (
            "Aviation risk is scored separately. This score does not validate factual accuracy, "
            "citation entailment, legal applicability, or overall operational safety. "
            "Do not treat 70 as automatic acceptance."
        ),
    }


# Original Sunayana evidence checklist retained unchanged.
def build_evidence_checklist(incident: dict[str, Any], retrieved: list[dict] | None = None) -> list[dict]:
    """Describe evidence supplied to the *system*; no claim of absence in the world."""
    text = str(incident.get("narrative", ""))
    source_text = " ".join(str(r.get("text", "")) for r in (retrieved or []))
    combined = (text + " " + source_text).lower()
    checks = [
        ("event_date", ["date", "2023", "2024", "2025", "2026"]),
        ("airport_runway", ["runway", "airport", "taxiway"]),
        ("crossing_clearance", ["crossing clearance", "cross runway", "cleared to cross"]),
        ("atc_communications", ["controller", "readback", "radio", "atc", "clearance"]),
        ("aircraft_positions", ["ads-b", "asde-x", "position", "surveillance", "track"]),
        ("readback_hearback", ["readback", "hearback", "acknowledged"]),
        ("airport_diagram", ["airport diagram", "taxi route", "figure", "marking"]),
        ("investigation_conclusions", ["probable cause", "contributing factors", "final report"]),
    ]
    result = []
    for name, tokens in checks:
        found = any(token in combined for token in tokens)
        status = "MENTIONED_IN_AVAILABLE_TEXT_UNVERIFIED" if found else "NOT_IDENTIFIED_IN_AVAILABLE_TEXT"
        if name == "event_date" and incident.get("event_date"):
            status = "USER_SUPPLIED_DATE_NOT_INDEPENDENTLY_VERIFIED"
        result.append({"item": name, "status": status})
    return result
