from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from blackwing.api import app, get_analysis_runner, get_repository, get_settings
from blackwing.agents.registry import AGENT_SPECS
from blackwing.agents.investigation.agent import remove_historical_case_evidence_ids
from blackwing.config import Settings
from blackwing.repository import MemoryRepository
from blackwing.schemas import (
    ComplianceOutput,
    CriticOutput,
    InvestigationOutput,
    MitigationOutput,
    RiskOutput,
    SimilarCaseOutput,
    VerificationOutput,
)


def test_runway_incursion_contract_vertical_slice() -> None:
    assert [spec.key for spec in AGENT_SPECS] == [
        "investigation", "similar_case", "compliance", "risk", "mitigation", "critic", "verification"
    ]
    assert all(spec.prompt_path.is_file() for spec in AGENT_SPECS)
    repo = MemoryRepository()
    captured: dict[str, Any] = {}

    async def fake_runner(narrative: str, settings: Settings, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "case_id": kwargs["case_id"],
            "status": "PRELIMINARY_REQUIRES_HUMAN_REVIEW",
            "workflow": "runway_incursion_investigation",
            "model": settings.ollama_model,
            "agent_outputs": {
                "investigation": InvestigationOutput(incident_summary="Mock runway incursion.").model_dump(mode="json"),
                "similar_case": SimilarCaseOutput(assessment="No relevant match in mock data.").model_dump(mode="json"),
                "compliance": ComplianceOutput(overall_applicability="requires_human_verification").model_dump(mode="json"),
                "risk": RiskOutput(risk_notes="Provisional mock assessment.").model_dump(mode="json"),
                "mitigation": MitigationOutput().model_dump(mode="json"),
                "critic": CriticOutput().model_dump(mode="json"),
                "verification": VerificationOutput().model_dump(mode="json"),
            },
            "trace": [
                {"step": role, "agent": role.title(), "model": settings.ollama_model, "status": "complete"}
                for role in ("investigation", "similar_case", "compliance", "risk", "mitigation", "critic", "verification")
            ],
            "limitations": ["Mock runner used for API contract test."],
        }

    settings = Settings(
        host="127.0.0.1",
        port=8000,
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen3:8b",
        ollama_timeout_seconds=1,
        ollama_temperature=0,
        ollama_think=False,
        ollama_max_output_tokens=512,
        api_base_url="",
        api_model="",
        api_key="",
        api_timeout_seconds=1,
        api_temperature=0,
        api_max_output_tokens=512,
        ui_api_url="http://127.0.0.1:8000",
        ui_request_timeout_seconds=1,
        rag_embedding_model="BAAI/bge-small-en-v1.5",
    )
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_analysis_runner] = lambda: fake_runner
    try:
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            openapi = client.get("/api/v1/openapi.json").json()
            assert "/api/v1/cases/{case_id}/runs" in openapi["paths"]
            assert "/api/v1/knowledge/search" in openapi["paths"]
            assert "ReportResponse" in openapi["components"]["schemas"]
            capabilities = client.get("/api/v1/capabilities").json()
            assert capabilities["incident_categories"] == [{"id": "runway_incursion", "status": "supported"}]

            document_response = client.post("/api/v1/knowledge/documents", json={
                "title": "Runway incursion procedure",
                "document_type": "procedure",
                "source_id": "PROC-1",
                "source_url": "https://example.org/procedure",
                "jurisdiction": "US",
                "document_date": "2026-01-01",
                "effective_date": "2026-01-01",
                "excerpt": "Procedure excerpt concerning runway incursion response.",
                "content": "Runway incursion prevention procedure, runway entry, and vehicle movement controls.",
            })
            assert document_response.status_code == 201
            search_response = client.post("/api/v1/knowledge/search", json={
                "query": "runway incursion vehicle entry",
                "document_types": ["procedure"],
                "jurisdiction": "US",
                "incident_date": "2026-10-30",
                "limit": 5,
            })
            assert search_response.status_code == 200
            assert search_response.json()["results"][0]["source_id"] == "PROC-1"

            case_response = client.post("/api/v1/cases", json={
                "category": "runway_incursion",
                "narrative": "A vehicle entered the runway as an aircraft taxied for departure.",
                "incident_date": "2026-10-30",
                "jurisdiction": "US",
                "location": "Example airport, runway 09",
            })
            assert case_response.status_code == 201
            case = case_response.json()
            case_id = case["case_id"]
            assert client.get("/api/v1/cases?status=draft").json()["total"] == 1

            run_response = client.post(f"/api/v1/cases/{case_id}/runs", json={
                "workflow": "runway_incursion_investigation",
                "include_retrieval": True,
            })
            assert run_response.status_code == 202
            run_id = run_response.json()["run_id"]
            assert client.get(f"/api/v1/runs/{run_id}").json()["status"] == "completed"
            assert client.get(f"/api/v1/cases/{case_id}/runs").json()[0]["run_id"] == run_id
            assert captured["retrieval_count"] >= 1
            assert "PROC-1" in {row["source_id"] for row in captured["evidence"]}

            report = client.get(f"/api/v1/cases/{case_id}/report")
            assert report.status_code == 200
            assert report.json()["is_final"] is False
            export = client.get(f"/api/v1/cases/{case_id}/report/export?format=json")
            assert export.status_code == 200
            assert "attachment" in export.headers["content-disposition"]
            assert set(report.json()["report"]["agent_outputs"]) == {
                "investigation", "similar_case", "compliance", "risk", "mitigation", "critic", "verification"
            }

            trace = client.get(f"/api/v1/runs/{run_id}/trace")
            assert trace.status_code == 200
            assert len(trace.json()["events"]) == 7

            review = client.post(f"/api/v1/cases/{case_id}/reviews", json={
                "reviewer_id": "reviewer-1",
                "decision": "request_changes",
                "rationale": "Please verify the source applicability and date.",
                "overrides": [{
                    "path": "compliance.applicability",
                    "action": "edit",
                    "replacement_value": "Requires human verification",
                    "rationale": "The source has not been independently assessed.",
                }],
            })
            assert review.status_code == 201
            assert client.get(f"/api/v1/cases/{case_id}/reviews").json()[0]["decision"] == "request_changes"
    finally:
        app.dependency_overrides.clear()


def test_historical_report_cannot_become_current_case_evidence() -> None:
    output = {
        "hypotheses": [{
            "text": "Controller awareness may have contributed.",
            "evidence_state": "hypothesis",
            "evidence_ids": ["ntsb-austin:0"],
            "source_ids": [],
            "rationale": None,
        }],
        "documented_facts": [],
        "reported_observations": [],
        "contributing_factors": [],
        "timeline_events": [],
    }
    sanitized = remove_historical_case_evidence_ids(output, [{
        "document_type": "incident_report",
        "document_id": "ntsb-austin",
        "chunk_id": "ntsb-austin:1",
        "chunk_index": 0,
        "source_id": "NTSB-DCA23FA149",
    }])
    claim = sanitized["hypotheses"][0]
    assert claim["evidence_ids"] == []
    assert claim["source_ids"] == ["NTSB-DCA23FA149"]
    assert claim["evidence_state"] == "hypothesis"
    assert "does not establish this factor" in claim["rationale"]
