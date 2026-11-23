"""Coverage for synthetic unseen-case evidence and decomposed report assembly."""
import asyncio

from blackwing.agents.report.schema import REPORT_TASK_ORDER
from blackwing.agents.report import workflow as report_workflow
from blackwing.case_inputs import synthetic_accounts
from blackwing.planning import plan_investigation


def _sample_unseen_case():
    case_id = "BW-UNSEEN-001"
    narrative = """CASE ID: BW-UNSEEN-001
ORIGINAL FICTIONAL training scenario. A maintenance sketch lacks a scale. No event date or country/jurisdiction is supplied.

KNOWN INPUT LIMITS
Event date: not supplied. Country/jurisdiction: not supplied.

SUPPLIED SYNTHETIC ACCOUNTS
BW-UNSEEN-001-SIM-E1: Driver reports stopping before the access limit.
BW-UNSEEN-001-SIM-E2: Crew reports apparent proximity, not a measured crossing.
CASE-SPECIFIC VERIFICATION FOCUS
Obtain surveyed boundaries and synchronized positions; distinguish proximity from entry.
"""
    cards = synthetic_accounts(narrative, case_id)
    evidence = [
        {"evidence_id": f"CASE-NARRATIVE:{case_id}", "source_id": f"CASE-NARRATIVE:{case_id}",
         "source_type": "user_submitted", "evidence_role": "reported",
         "evidence_status": "USER_REPORTED_UNVERIFIED", "excerpt": narrative},
        *cards,
    ]
    investigation = {
        "incident_summary": "Two supplied accounts describe the van's reported position differently.",
        # Simulate a small model returning only one of two supplied accounts.
        "reported_observations": [{"text": cards[0]["excerpt"],
            "evidence_state": "reported_observation", "evidence_ids": [cards[0]["evidence_id"]]}],
        "unknowns": ["The vehicle's surveyed position and synchronized time are unknown."],
    }
    outputs = {
        "investigation": investigation,
        "similar_case": {"case_status": "unseen_case_no_historical_match", "matches": [],
                         "assessment": "No exact historical match was retrieved."},
        "compliance": {"overall_applicability": "not_assessed", "checks": [], "missing_context": []},
        "risk": {"assessment_status": "unassessed_insufficient_inputs", "hazards": [],
                 "risk_notes": "Severity and likelihood lack verified support."},
        "mitigation": {"recommendations": []},
        "critic": {"issues": []},
        "verification": {"claims_needing_source": ["Obtain surveyed boundaries and synchronized positions."],
                         "conflicts": [], "completeness_status": "incomplete"},
    }
    return case_id, narrative, evidence, cards, outputs


def test_unseen_plan_uses_task_decomposed_report():
    plan = plan_investigation({"narrative": "A fictional synthetic training case.", "evidence": []})
    assert plan["task_modes"]["report"] == "task_decomposed"


def test_deterministic_report_preserves_every_supplied_synthetic_account():
    _, narrative, evidence, cards, outputs = _sample_unseen_case()
    result, _ = asyncio.run(report_workflow.create_report(
        client=None, settings=None, narrative=narrative, evidence=evidence,
        prior_outputs=outputs, deterministic=True,
    ))
    overview = result["sections"][0]
    reported = {point["text"]: point for point in overview["key_points"]}
    for card in cards:
        point = reported[card["excerpt"]]
        assert card["evidence_id"] in point["evidence_ids"]
        assert point["classification"] == "reported_observation"
    assert sum(point["text"] == cards[0]["excerpt"] for point in overview["key_points"]) == 1
    assert any("maintenance sketch lacks a scale" in point["text"].lower()
               for point in overview["key_points"])
    assert any("event date: not supplied" in point["text"].lower()
               for point in overview["key_points"])
    assert any(point["classification"] == "verification_gap"
               and "surveyed position" in point["text"] for point in result["sections"][1]["key_points"])
    assert any(point["classification"] == "verification_gap"
               and "surveyed boundaries" in point["text"] for point in result["sections"][3]["key_points"])
    findings = result["sections"][1]
    for card in cards:
        assert any(point["text"] == card["excerpt"]
                   and card["evidence_id"] in point["evidence_ids"]
                   for point in findings["key_points"])


def test_task_report_calls_each_section_and_restores_omitted_accounts(monkeypatch):
    _, narrative, evidence, cards, outputs = _sample_unseen_case()
    calls = []

    async def fake_invoke(spec, **kwargs):
        calls.append(spec.key)
        return {"summary": "Concise section draft.", "key_points": []}

    monkeypatch.setattr(report_workflow, "invoke_agent", fake_invoke)
    result, trace = asyncio.run(report_workflow.create_report(
        client=None, settings=None, narrative=narrative, evidence=evidence,
        prior_outputs=outputs,
    ))
    assert calls == [f"report_{task_id}" for task_id in REPORT_TASK_ORDER]
    assert [section["task_id"] for section in result["sections"]] == list(REPORT_TASK_ORDER)
    overview = result["sections"][0]
    points = {point["text"]: point for point in overview["key_points"]}
    for card in cards:
        assert card["excerpt"] in points
        assert card["evidence_id"] in points[card["excerpt"]]["evidence_ids"]
    assert trace[0]["status"] == "fixed_validated_plan"
    assert all(section["generation_method"] == "model" for section in result["sections"])


def test_failed_report_task_falls_back_without_dropping_accounts(monkeypatch):
    _, narrative, evidence, cards, outputs = _sample_unseen_case()

    async def failed_invoke(spec, **kwargs):
        raise RuntimeError("mocked report task failure")

    monkeypatch.setattr(report_workflow, "invoke_agent", failed_invoke)
    result, trace = asyncio.run(report_workflow.create_report(
        client=None, settings=None, narrative=narrative, evidence=evidence,
        prior_outputs=outputs,
    ))
    overview = result["sections"][0]
    for card in cards:
        assert any(point["text"] == card["excerpt"]
                   and card["evidence_id"] in point["evidence_ids"]
                   for point in overview["key_points"])
    assert all(section["generation_method"] == "deterministic_fallback"
               for section in result["sections"])
    assert all(item["status"] == "deterministic_fallback" for item in trace[1:])


def test_empty_model_section_summary_uses_case_specific_source_summary(monkeypatch):
    _, narrative, evidence, cards, outputs = _sample_unseen_case()

    async def empty_summary(spec, **kwargs):
        return {"summary": "", "key_points": []}

    monkeypatch.setattr(report_workflow, "invoke_agent", empty_summary)
    result, _ = asyncio.run(report_workflow.create_report(
        client=None, settings=None, narrative=narrative, evidence=evidence,
        prior_outputs=outputs,
    ))
    findings = result["sections"][1]
    assert all(card["excerpt"] in findings["summary"] for card in cards)
    assert "probable cause" in findings["summary"].lower()


def test_report_merge_deduplicates_repeated_text_and_unions_citations():
    section = {"key_points": [
        {"text": "Same supplied observation", "classification": "verification_gap",
         "evidence_ids": [], "source_ids": []},
        {"text": "Same supplied observation", "classification": "reported_observation",
         "evidence_ids": ["SIM-E1"], "source_ids": []},
    ]}
    merged = report_workflow._merge_source_detail(section, "overview", {}, [], "")
    points = [point for point in merged["key_points"] if point["text"] == "Same supplied observation"]
    assert len(points) == 1
    assert points[0]["classification"] == "reported_observation"
    assert points[0]["evidence_ids"] == ["SIM-E1"]
