from test_streamlit_integration import backend
from blackwing.report_quality import assess_quality


def sample_report(*, with_citations=True):
    return {
        "agent_outputs": {
            "investigation": {"documented_facts": [{"fact": "runway entry", "evidence_ids": ["BW-1"] if with_citations else []}], "unknowns": ["separation unknown"]},
            "similar_case": {},
            "compliance": {"applicability_status": "requires_human_verification"},
            "risk": {"hazards": [{"hazard_id": "H1", "scoring_status": "unassessed_insufficient_inputs"}]},
            "mitigation": {"recommendations": [{"hazard_ids": ["H1"]}]},
            "critic": {"issues": [] , "review_status": "completed"},
            "verification": {"human_approval_required": True},
        },
        "generated_report": {
            "executive_summary": "Narrative summary",
            "sections": [
                {"key_points": [{"classification": "documented_fact", "source_ids": ["BW-1"]}]},
                {"key_points": []},
                {"key_points": []},
                {"key_points": []},
            ]
        }
    }


def test_prompt_quality_separate():
    assert assess_quality({}, None)["score"] == 0
    prompt = {"narrative": "An aircraft entered a runway without clearance", "location": "KJFK", "incident_date": "2023-01-13", "jurisdiction": "US", "source_incident_id": "NTSB-ID"}
    assert assess_quality(prompt)["score"] == 100


def test_report_that_meets_stated_checks_gets_over_70():
    report = sample_report()
    r = assess_quality({}, report)
    assert r["score"] >= 70
    assert sum(c["weight"] for c in r["checks"]) == 100


def test_missing_citations_are_penalized_not_vacuously_accepted():
    report = sample_report(with_citations=False)
    r = assess_quality({}, report)
    assert r["score"] <= 59
    assert any("citation" in f.lower() for f in r["flags"])


def test_empty_report_does_not_get_free_reference_points():
    r = assess_quality({}, {"agent_outputs": {}, "generated_report": {}})
    assert r["score"] < 70
    assert next(c for c in r["checks"] if c["check"] == "Report factual points referenced")["earned"] == 0


def test_safe_unknowns_not_confused_with_cited_facts():
    r = sample_report()
    r["agent_outputs"]["investigation"]["documented_facts"] = []
    result = assess_quality({}, r)
    assert result["score"] < 70
    assert next(c for c in result["checks"] if c["check"] == "Unknowns/completeness disclosed")["earned"] == 7


def test_critic_issues_are_not_hidden_or_treated_as_no_review():
    r = sample_report()
    r["agent_outputs"]["critic"]["issues"] = ["Citation incomplete"]
    result = assess_quality({}, r)
    assert next(c for c in result["checks"] if c["check"] == "Critic review and issue handling")["earned"] == 2


def test_unscored_risk_with_explanation_is_acceptable():
    r = sample_report()
    score = assess_quality({}, r)
    assert next(c for c in score["checks"] if c["check"] == "Risk rationale or explicit abstention")["earned"] == 8


def test_unsupported_risk_index_does_not_earn_risk_points():
    r = sample_report()
    r["agent_outputs"]["risk"]["hazards"][0] = {"hazard_id": "H1", "risk_index": "C3", "risk_level": "medium", "evidence_ids": []}
    score = assess_quality({}, r)
    assert next(c for c in score["checks"] if c["check"] == "Risk rationale or explicit abstention")["earned"] == 0


def test_report_score_does_not_import_prompt_score():
    report = sample_report()
    empty = assess_quality({}, report)
    full = assess_quality({"narrative": "Complete supplied incident narrative", "location": "Airport", "incident_date": "2026-01-01", "jurisdiction": "IN", "source_incident_id": "ID"}, report)
    assert empty == full
    assert assess_quality({})["score"] == 0


def test_backend_compliance_and_risk_aliases_receive_documentation_credit():
    report = sample_report()
    report["agent_outputs"]["compliance"] = {"overall_applicability": "not_assessed", "checks": [], "missing_context": ["Applicable source edition unavailable"]}
    report["agent_outputs"]["risk"] = {"assessment_status": "unassessed_insufficient_inputs", "hazards": []}
    result = assess_quality({}, report)
    assert next(c for c in result["checks"] if c["check"] == "Regulatory applicability review recorded")["earned"] == 8
    assert next(c for c in result["checks"] if c["check"] == "Risk rationale or explicit abstention")["earned"] == 8


def test_null_and_blank_ids_are_not_citations_but_valid_alternate_ids_are():
    report = sample_report()
    fact = report["agent_outputs"]["investigation"]["documented_facts"][0]
    fact["evidence_ids"] = [None, " ", ""]
    assert assess_quality({}, report)["score"] <= 59
    fact["source_ids"] = ["ACTUAL-SOURCE-ID"]
    assert next(c for c in assess_quality({}, report)["checks"] if c["check"] == "Factual evidence references present")["earned"] == 13


def test_grade_has_no_forced_minimum_and_discloses_integrity_caps():
    report = sample_report(with_citations=False)
    result = assess_quality({}, report)
    assert result["score"] <= 59
    assert result["raw_score"] == sum(c["earned"] for c in result["checks"])
    assert result["flags"] and result["rubric_version"] == "2"
    assert assess_quality({}, {})["score"] == 0


def test_risk_category_magnitude_never_changes_document_completeness():
    report = sample_report()
    hazard = report["agent_outputs"]["risk"]["hazards"][0]
    hazard.update(scoring_status="provisional_reference_only", risk_index="C3", risk_level="medium", severity=3, likelihood="C",
                  evidence_ids=["BW-1"], severity_basis="Recorded basis", likelihood_basis="Recorded basis")
    before = assess_quality({}, report)
    hazard.update(severity=1, likelihood="A", risk_index="A1", risk_level="high")
    assert assess_quality({}, report) == before


def test_assessment_does_not_mutate_incident_or_report():
    from copy import deepcopy
    incident, report = {"narrative": "Synthetic incident supplied by the user"}, sample_report()
    original = deepcopy((incident, report))
    assess_quality(incident)
    assess_quality(incident, report)
    assert (incident, report) == original


def test_report_evaluator_reads_sqlite_without_running_recovery(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    from blackwing.repository import SQLiteRepository
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_report_completeness.py"
    spec = importlib.util.spec_from_file_location("offline_quality_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    db = tmp_path / "cases.sqlite3"
    repo = SQLiteRepository(db)
    repo.create_case("CASE", {"narrative": "A synthetic incident with limited supplied intake", "category": "runway_incursion"})
    repo.create_run("ACTIVE", "CASE", "runway_incursion_investigation", "mock")
    repo.close()
    report = sample_report()
    report["case_id"] = "CASE"
    exported = tmp_path / "report.json"
    exported.write_text(json.dumps({"run_id": "SAVED", "report": report}), encoding="utf-8")
    import sqlite3
    with sqlite3.connect(db) as connection:
        before = connection.execute("SELECT payload FROM records WHERE kind='runs' AND id='ACTIVE'").fetchone()[0]
    result = module.evaluate([exported], db)
    with sqlite3.connect(db) as connection:
        after = connection.execute("SELECT payload FROM records WHERE kind='runs' AND id='ACTIVE'").fetchone()[0]
    assert before == after
    assert result["reports"][0]["prompt_completeness"]["score"] == 25
    assert result["database_access"] == "read-only"


def test_offline_eval_does_not_infer_missing_intake_from_generated_summary(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_report_completeness.py"
    spec = importlib.util.spec_from_file_location("offline_quality_missing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exported = tmp_path / "report.json"
    exported.write_text(json.dumps(sample_report()), encoding="utf-8")
    result = module.evaluate([exported])
    assert result["reports"][0]["prompt_completeness"] is None
    assert "not inferred" in result["reports"][0]["prompt_source"]


def test_ui_displays_both_independent_scores_and_preserves_workspace(backend):
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    from test_streamlit_integration import prepared_case
    case, _ = prepared_case()
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "blackwing/ui.py"), default_timeout=15)
    at.session_state["workspace_view"] = "case"
    at.session_state["workspace_case"] = case["incident"]["case_id"]
    at.run()
    assert not at.exception and len(at.tabs) == 3
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Prompt Completeness"] == str(assess_quality(case["incident"])["score"]) + "%"
    assert metrics["Report Completeness"] == str(assess_quality(case["incident"], case["report"])["score"]) + "%"
    assert "Aviation risk" in metrics
    assert len([df for df in at.dataframe if {"weight", "earned"} <= set(df.value.columns)]) == 2
    assert any("source-branch checklist" in caption.value for caption in at.caption)
