"""Offline tests: no hosted calls, Ollama, model downloads or RAG index needed."""
from __future__ import annotations

from blackwing.agents.evidence_utils import citation_map, is_direct, resolve
from blackwing.agents.investigation.grounding import ground_investigation_output
from blackwing.agents.investigation.agent import remove_historical_case_evidence_ids
from blackwing.agents.similar_case.agent import build_output
from blackwing.agents.mitigation.grounding import ground_mitigation_output
from blackwing.agents.critic.grounding import audit_critic_output
from blackwing.agents.verification.grounding import ground_verification_output
from blackwing.reporting import render_markdown_report


CURRENT = {"chunk_id": "DCA23FA149:5", "document_id": "DCA23FA149", "chunk_index": 4,
           "source_id": "NTSB-AUS", "source_type": "official_factual", "evidence_role": "case_factual",
           "incident_id": "DCA23FA149", "page_number": 5, "excerpt": "Current-case factual passage."}
HISTORY = {"chunk_id": "OLD:2", "source_id": "NTSB-OLD", "document_id": "OLD",
           "incident_id": "DCA22FA123", "source_type": "official_final", "document_type": "incident_report",
           "evidence_role": "historical_analogy", "title": "Historic taxi clearance conflict",
           "excerpt": "A runway taxi clearance conflict occurred in earlier case."}
UPLOAD = {"evidence_id": "UPLOAD:1", "source_id": "UPLOAD", "source_type": "user_submitted",
          "evidence_role": "reported", "incident_id": "DCA23FA149"}


def claim(text="Clearance was issued", refs=None, state="documented_fact"):
    return {"text": text, "evidence_state": state, "evidence_ids": refs or [],
            "source_ids": ["fabricated-source"], "rationale": None}


def test_citation_alias_preserves_current_fact_and_drops_history():
    raw = {"incident_summary": "A synthetic runway conflict was reported.",
           "documented_facts": [claim(refs=["DCA23FA149:4", "OLD:2", "fake"])],
           "reported_observations": [], "hypotheses": [], "contributing_factors": [],
           "timeline_events": [{"event": "Tower issued clearance", "evidence_ids": ["DCA23FA149:5"],
                                "evidence_state": "documented_fact"}], "unknowns": []}
    out = ground_investigation_output(raw, [CURRENT, HISTORY], incident_id="DCA23FA149")
    assert out["documented_facts"][0]["evidence_ids"] == ["DCA23FA149:5"]
    assert out["documented_facts"][0]["source_ids"] == ["NTSB-AUS"]
    assert out["timeline_events"][0]["evidence_state"] == "reported_observation"
    assert "Invalid" in out["documented_facts"][0]["rationale"]


def test_other_incident_and_no_source_demoted():
    old = dict(CURRENT, incident_id="OTHER")
    out = ground_investigation_output({"documented_facts": [claim(refs=["DCA23FA149:5"])]}, [old],
                                      incident_id="DCA23FA149")
    assert out["documented_facts"][0]["evidence_state"] == "unknown"
    assert out["documented_facts"][0]["evidence_ids"] == []
    assert out["unknowns"]


def test_user_report_never_becomes_official_fact():
    out = ground_investigation_output({"documented_facts": [claim(refs=["UPLOAD:1"])]}, [UPLOAD],
                                      incident_id="DCA23FA149")
    assert out["documented_facts"][0]["evidence_state"] == "reported_observation"


def test_original_legacy_historical_test_behavior():
    old = {"chunk_id": "ntsb-austin:1", "chunk_index": 0, "document_id": "ntsb-austin",
           "source_id": "NTSB-DCA23FA149", "document_type": "incident_report"}
    out = remove_historical_case_evidence_ids({"hypotheses": [claim("Maybe a factor", ["ntsb-austin:0"],
                                                                      state="hypothesis")]}, [old])
    assert out["hypotheses"][0]["evidence_ids"] == []
    assert "NTSB-DCA23FA149" in out["hypotheses"][0]["source_ids"]
    assert "does not establish this factor" in out["hypotheses"][0]["rationale"]


def test_similar_case_deduplicates_and_excludes_current():
    record = dict(HISTORY, historical_findings=["Historical report's own finding"],
                  comparison_notes=["Weather comparison unknown"], incident_date="2022-02-03")
    current = dict(CURRENT, document_type="incident_report", title="Current runway event")
    out = build_output("Runway taxi clearance conflict", [record, dict(record), current],
                       current_incident_id="DCA23FA149")
    assert len(out["matches"]) == 1
    assert out["matches"][0]["source_id"] == "NTSB-OLD"
    assert "analogy" in out["assessment"].lower()
    assert "Historical report's own finding" in out["matches"][0]["source_findings"]


def test_similar_no_match_does_not_invent():
    out = build_output("Runway conflict", [], current_incident_id="DCA23FA149")
    assert not out["matches"]
    assert out["case_status"] == "unseen_case_no_historical_match"


def test_mitigation_links_only_actual_hazard_and_marks_unverified():
    priors = {"risk": {"hazards": [{"hazard_id": "HAZ-001", "evidence_ids": []}]},
              "investigation": {"contributing_factors": []}}
    raw = {"recommendations": [
        {"action": "Inspect holding-position signage", "hazard_ids": ["HAZ-001", "FAKE"],
         "priority": "high", "responsible_area": "Regulator"},
        {"action": "Upgrade airport runway", "hazard_ids": ["FAKE"], "priority": "critical"}],
        "assumptions": []}
    out = ground_mitigation_output(raw, priors)
    assert len(out["recommendations"]) == 1
    assert out["recommendations"][0]["hazard_ids"] == ["HAZ-001"]
    assert out["recommendations"][0]["priority"] == "proposed_requires_evidence"
    assert out["recommendations"][0]["responsible_area"] is None
    assert "Omitted recommendation" in " ".join(out["assumptions"])


def test_critic_flags_citation_and_regulatory_gaps():
    prior = {"investigation": {"documented_facts": [claim(refs=[])]},
             "risk": {"hazards": [{"hazard_id": "HAZ-001", "evidence_ids": []}]},
             "compliance": {"overall_applicability": "not_assessed"},
             "mitigation": {"recommendations": []}}
    out = audit_critic_output({"issues": []}, prior)
    types = {i["issue_type"] for i in out["issues"]}
    assert "ungrounded_fact" in types
    assert "ungrounded_hazard" in types
    assert "regulatory_context_gap" in types
    assert out["overall_assessment"] == "requires_human_review"


def test_verification_never_approves_and_surfaces_problems():
    prior = {"investigation": {"documented_facts": [claim(refs=[])]},
             "risk": {"hazards": [{"hazard_id": "HAZ-001", "evidence_ids": []}]},
             "compliance": {"overall_applicability": "not_assessed"},
             "mitigation": {"recommendations": []}, "critic": {"issues": []}}
    out = ground_verification_output({"human_approval_required": False,
                                       "completeness_status": "complete"}, prior)
    assert out["human_approval_required"]
    assert out["completeness_status"] == "incomplete"
    assert "Clearance was issued" in out["claims_needing_source"]


def test_markdown_report_has_all_sections_citations_no_approval_and_redacts_contacts():
    payload = {"report": {"case_id": "TEST-1", "model": "qwen3:8b", "status": "PRELIMINARY_REQUIRES_HUMAN_REVIEW",
                          "agent_outputs": {
                              "investigation": {"incident_summary": "Witness a@example.com reported event.",
                                                "documented_facts": [claim(refs=["DCA23FA149:5"])]},
                              "similar_case": {"matches": []}, "compliance": {"checks": [], "overall_applicability": "not_assessed"},
                              "risk": {"hazards": [{"hazard_id": "HAZ-001", "description": "Potential conflict", "evidence_ids": []}],
                                       "assessment_status": "provisional"},
                              "mitigation": {"recommendations": []}, "critic": {"issues": []},
                              "verification": {"human_approval_required": True, "completeness_status": "incomplete"}},
                          "limitations": ["Human review required"]}}
    markdown = render_markdown_report(payload)
    assert "## 7. Quality Review" in markdown
    assert "DCA23FA149:5" in markdown
    assert "a@example.com" not in markdown
    assert "[EMAIL REDACTED]" in markdown
    assert "Human approval required:** Yes" in markdown


def test_only_valid_citations_are_resolved():
    assert resolve(["MISSING", "OLD:2"], [CURRENT, HISTORY], direct_only=True)[0] == []
    assert citation_map([CURRENT])["DCA23FA149:4"][0] == "DCA23FA149:5"
    assert is_direct(CURRENT, "DCA23FA149")
    assert not is_direct(HISTORY, "DCA23FA149")
