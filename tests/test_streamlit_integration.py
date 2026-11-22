"""Real Streamlit AppTest + real FastAPI contracts; no model/provider IO."""
import copy
import importlib
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

from blackwing import api, copilot
from blackwing.copilot_state import risk_display
from blackwing.copilot_report import render_report_pdf
from blackwing.report_quality import assess_quality, build_evidence_checklist
from blackwing.repository import SQLiteRepository
from blackwing.schemas import (InvestigationOutput, SimilarCaseOutput, ComplianceOutput, RiskOutput,
                                MitigationOutput, CriticOutput, VerificationOutput)
from test_agentic_local import settings

ROOT = Path(__file__).resolve().parents[1]


async def synthetic_runner(narrative, config, **kwargs):
    refs = [r["evidence_id"] for r in kwargs.get("evidence", []) if r.get("evidence_id")]
    return {"case_id": kwargs["case_id"], "status": "PRELIMINARY_REQUIRES_HUMAN_REVIEW",
            "workflow": "runway_incursion_investigation", "model": "mock-eight-agent-contract",
            "agent_outputs": {
                "investigation": InvestigationOutput(incident_summary="Synthetic runway conflict", documented_facts=[{
                    "text": "Aircraft crossed the runway", "evidence_ids": refs, "evidence_state": "documented_fact"}], unknowns=["Clearance sequence requires verification"]).model_dump(mode="json"),
                "similar_case": SimilarCaseOutput(assessment="No synthetic historical match").model_dump(mode="json"),
                "compliance": ComplianceOutput().model_dump(mode="json"),
                "risk": RiskOutput(hazards=[{"hazard_id": "H1", "description": "Runway conflict", "evidence_ids": refs,
                    "severity": 3, "likelihood": "C", "risk_index": "C3", "risk_level": "medium",
                    "severity_basis": "Synthetic basis", "likelihood_basis": "Synthetic basis",
                    "scoring_status": "provisional_reference_only" if refs else "unassessed_insufficient_inputs"}],
                    assessment_status="provisional_reference_matrix_requires_human_review" if refs else "unassessed_insufficient_inputs").model_dump(mode="json"),
                "mitigation": MitigationOutput(recommendations=[{"action": "Review coordination", "hazard_ids": ["H1"]}]).model_dump(mode="json"),
                "critic": CriticOutput(issues=[{"issue_type": "review_gap", "description": "Review clearance evidence"}]).model_dump(mode="json"),
                "verification": VerificationOutput().model_dump(mode="json")},
            "generated_report": {"executive_summary": "Synthetic preliminary report", "human_approval_required": True,
                "sections": [{"task_id": key, "title": key.replace("_", " "), "summary": "Synthetic source summary",
                              "generation_method": "deterministic_fallback", "key_points": []} for key in (
                                  "overview", "findings_and_causes", "comparisons_and_compliance", "risk_and_follow_up")]},
            "trace": [{"step": key, "status": "complete"} for key in (
                "investigation", "similar_case", "compliance", "risk", "mitigation", "critic", "verification", "report")],
            "limitations": ["Synthetic mocked model outputs; qualified review required"]}


@pytest.fixture
def backend(monkeypatch, tmp_path):
    repo = SQLiteRepository(tmp_path / "backend.sqlite3")
    api.app.dependency_overrides[api.get_repository] = lambda: repo
    api.app.dependency_overrides[api.get_settings] = settings
    api.app.dependency_overrides[api.get_analysis_runner] = lambda: synthetic_runner
    monkeypatch.setattr(copilot, "client", lambda: TestClient(api.app))
    # No actual index use is needed for API-schema/UI integration tests.
    monkeypatch.setattr(api.retriever, "search", lambda *args, **kw: {"results": []})
    try:
        yield repo
    finally:
        api.app.dependency_overrides.clear()
        repo.close()


def prepared_case():
    evidence = copilot.file_evidence([("factual.txt", b"Aircraft crossed the runway while taxiing.")], "Synthetic witness record")
    case = copilot.create_case(narrative="Synthetic runway conflict with a crossing aircraft.", location="Synthetic airport",
                               jurisdiction="IN", incident_date="2026-01-01", evidence=evidence)
    run = copilot.start_investigation(case["case_id"], include_retrieval=False)
    return copilot.get_case(case["case_id"]), run


def test_all_integration_modules_import_without_app_or_model_calls():
    for module in ("ui", "copilot", "copilot_state", "copilot_report", "copilot_ui", "report_quality", "cloud.streamlit_app"):
        assert importlib.import_module("blackwing." + module)


def test_api_bridge_full_contract_and_canonical_reviews(backend):
    case, run = prepared_case()
    assert case["report"]["model"] == "mock-eight-agent-contract"
    assert copilot.get_run(run["run_id"])["status"] == "completed"
    assert len([e for e in copilot.get_trace(run["run_id"])["events"] if e["step"] in {"investigation", "similar_case", "compliance", "risk", "mitigation", "critic", "verification", "report"}]) == 8
    assert copilot.get_audit(run["run_id"])["resumable"] is False
    fact = case["findings"][0]
    assert fact["status"] == "pending"
    copilot.review_finding(case, fact["finding_id"], "accept", "Human Reviewer", "Verified original passage")
    fresh = copilot.get_case(case["incident"]["case_id"])
    assert fresh["findings"][0]["status"] == "accepted"
    assert backend.list_reviews(case["incident"]["case_id"])[0]["overrides"][0]["path"].startswith("runs.")
    pdf = render_report_pdf(copilot.generate_report(fresh))
    assert pdf.startswith(b"%PDF-")
    text = " ".join(page.extract_text() for page in PdfReader(BytesIO(pdf)).pages)
    assert "Aircraft crossed the runway" in text and "Human Reviewer" in text
    assert copilot.export_report(case["incident"]["case_id"], "json").startswith(b"{")
    assert b"BLACKWING" in copilot.export_report(case["incident"]["case_id"], "md").upper()
    assert len(copilot.list_cases()) == 1


def test_review_does_not_transfer_to_a_new_run(backend):
    case, _ = prepared_case()
    copilot.review_finding(case, case["findings"][0]["finding_id"], "accept", "Reviewer", "Checked")
    copilot.start_investigation(case["incident"]["case_id"], include_retrieval=False)
    fresh = copilot.get_case(case["incident"]["case_id"])
    assert fresh["report_run_id"] != case["report_run_id"]
    assert fresh["findings"][0]["status"] == "pending"
    assert copilot.generate_report(fresh)["accepted_count"] == 0


@pytest.mark.parametrize("risk", [{}, {"hazards": []}, {"hazards": [{"hazard_id": "H1", "severity": 3, "likelihood": "C", "risk_index": "C3", "risk_level": "medium", "scoring_status": "unassessed_insufficient_inputs"}]}])
def test_missing_or_unassessed_risk_is_never_calculated(risk):
    display = risk_display({"agent_outputs": {"risk": risk}})
    assert "FAA AC" in display["framework"] and "not a binding" in display["notice"]
    assert all(row["Risk index"] == "Unassessed" for row in display["rows"])


def test_backend_classification_is_displayed_without_recalculation(backend):
    case, _ = prepared_case()
    report = case["report"]
    display = risk_display(report)
    assert display["rows"][0]["Severity"].startswith("3 — major")
    assert display["rows"][0]["Risk index"] == "C3"
    assert display["rows"][0]["Classification"] == "medium"
    # Confirm rendering trusts the backend class rather than inventing a product.
    report["agent_outputs"]["risk"]["hazards"][0]["risk_level"] = "backend-supplied"
    assert risk_display(report)["rows"][0]["Classification"] == "backend-supplied"


def test_quality_checklist_preserved_and_quality_independent_of_risk(backend):
    case, _ = prepared_case()
    report = case["report"]
    quality = assess_quality(case["incident"], report)
    assert any(c["check"] == "Critic review and issue handling" and c["earned"] == 2 for c in quality["checks"])
    original = quality["score"]
    changed = copy.deepcopy(report)
    hazard = changed["agent_outputs"]["risk"]["hazards"][0]
    hazard.update(severity=1, likelihood="A", risk_index="A1", risk_level="high")
    assert assess_quality(case["incident"], changed)["score"] == original
    missing = assess_quality({"narrative": "Short"})
    assert missing["score"] == 0 and len(missing["missing_information"]) == 5
    checklist = build_evidence_checklist({"narrative": "ATC readback runway crossing", "event_date": "2026-01-01"})
    assert len(checklist) == 8
    assert checklist[0]["status"] == "USER_SUPPLIED_DATE_NOT_INDEPENDENTLY_VERIFIED"


@pytest.mark.parametrize("filename,data", [("x.txt", b""), ("x.bin", b"arbitrary"), ("x.txt", b"x"*20001)])
def test_upload_validation_prevents_silent_truncation(filename, data):
    with pytest.raises(ValueError):
        copilot.file_evidence([(filename, data)], "Provenance")


def test_streamlit_create_upload_run_review_and_exports(backend):
    at = AppTest.from_file(str(ROOT / "blackwing/ui.py"), default_timeout=15).run()
    assert not at.exception
    at.button(key="home-new").click().run()
    at.text_input(key="intake-location").set_value("Synthetic airport")
    at.text_input(key="intake-jurisdiction").set_value("IN")
    at.text_area(key="intake-narrative").set_value("Synthetic runway crossing involving a taxiing aircraft.")
    at.file_uploader(key="intake-files").upload("witness.txt", b"Aircraft crossed the runway while taxiing.")
    at.text_input(key="intake-provenance").set_value("Synthetic source")
    at.button(key="create-investigation").click().run()
    assert not at.exception
    assert len(backend.cases) == 1 and not backend.runs
    assert backend.list_cases(None, 100, 0)[0][0]["evidence"]
    at.button(key="run-investigation").click().run()
    assert not at.exception and len(backend.runs) == 1
    assert len(at.tabs) == 9
    risk_frames = [df.value for df in at.dataframe if "Risk index" in df.value.columns]
    assert risk_frames[0].iloc[0]["Risk index"] == "C3"
    assert any("not a binding" in w.value for w in at.warning)
    at.text_input(key="reviewer-name").set_value("Reviewer")
    at.text_area(key="review-rationale").set_value("Verified actual source text")
    at.checkbox(key="review-confirmed").check()
    at.button(key="accept-finding").click().run()
    assert not at.exception
    assert at.get_by_key("export-pdf")
    # A rerun never triggers another investigation or creates another case DB.
    at.run()
    assert len(backend.runs) == 1 and len(backend.cases) == 1


def test_cloud_entrypoint_uses_same_backend_and_no_temp_database(backend):
    at = AppTest.from_file(str(ROOT / "blackwing/cloud/streamlit_app.py"), default_timeout=15).run()
    assert not at.exception
    assert any("same backend" in caption.value for caption in at.caption)
    assert "cloud_case_directory" not in at.session_state
    assert not backend.cases


def test_streamlit_missing_risk_and_intake_quality(backend):
    at = AppTest.from_file(str(ROOT / "blackwing/ui.py"), default_timeout=15).run()
    at.button(key="home-new").click().run()
    at.button(key="check-intake").click().run()
    assert not at.exception
    assert any(m.label == "Prompt Completeness" and m.value == "0%" for m in at.metric)
    at.text_area(key="intake-narrative").set_value("Synthetic runway crossing with missing evidence.")
    at.button(key="create-investigation").click().run()
    assert not at.exception
    assert any("Unassessed" in alert.value for alert in at.info)


def test_backend_history_survives_new_frontend_session(backend):
    case, _ = prepared_case()
    at = AppTest.from_file(str(ROOT / "blackwing/ui.py"), default_timeout=15).run()
    at.button(key="home-history").click().run()
    assert not at.exception
    assert case["incident"]["case_id"] in at.selectbox(key="saved-case").options or len(at.selectbox(key="saved-case").options) == 1
    at.button(key="resume-case").click().run()
    assert not at.exception and len(backend.runs) == 1


def test_corrupt_pdf_reports_a_controlled_upload_error():
    with pytest.raises(ValueError, match="could not be read"):
        copilot.file_evidence([("broken.pdf", b"invalid PDF data")], "Provenance")


def test_valid_pdf_upload_extracts_actual_text_without_a_model(backend):
    case, _ = prepared_case()
    copilot.review_finding(case, case["findings"][0]["finding_id"], "accept", "Reviewer", "Checked original")
    fresh = copilot.get_case(case["incident"]["case_id"])
    pdf = render_report_pdf(copilot.generate_report(fresh))
    evidence = copilot.file_evidence([("reviewed.pdf", pdf)], "PDF source")
    assert "Aircraft crossed the runway" in evidence[0]["excerpt"]
    assert "FAA AC 150/5200-37A" in evidence[0]["excerpt"]


def test_stale_run_risk_and_review_are_not_presented_as_current(backend):
    case, _ = prepared_case()
    backend.create_run("failed-new-run", case["incident"]["case_id"], "runway_incursion_investigation", "mock")
    backend.patch_run("failed-new-run", {"status": "failed", "error_code": "interrupted", "error": "Synthetic interruption"})
    backend.update_case(case["incident"]["case_id"], {"status": "analysis_failed"})
    fresh = copilot.get_case(case["incident"]["case_id"])
    with pytest.raises(ValueError, match="newer run"):
        copilot.review_finding(fresh, fresh["findings"][0]["finding_id"], "accept", "Reviewer", "Checked")
    at = AppTest.from_file(str(ROOT / "blackwing/ui.py"), default_timeout=15)
    at.session_state["workspace_view"] = "case"
    at.session_state["workspace_case"] = case["incident"]["case_id"]
    at.run()
    assert not at.exception
    assert any("earlier completed report" in warning.value for warning in at.warning)
    assert any("interrupted" in warning.value for warning in at.warning)
