"""Real LangGraph orchestration, fake model responses; no API usage.

Run on a laptop after dependencies installed. pytest.importorskip keeps
logic-only tests independently runnable when LangGraph is not installed.
"""
import asyncio

import pytest

pytest.importorskip("langgraph")
from blackwing.config import Settings
from blackwing import service


def _settings():
    return Settings(host="127.0.0.1", port=8001, llm_provider="ollama",
        ollama_base_url="http://localhost:11434", ollama_model="qwen3:8b",
        ollama_timeout_seconds=2, ollama_temperature=0, ollama_think=False,
        ollama_max_output_tokens=512, api_base_url="", api_model="", api_key="",
        api_timeout_seconds=2, api_temperature=0, api_max_output_tokens=512,
        ui_api_url="http://127.0.0.1:8001", ui_request_timeout_seconds=2,
        rag_embedding_model="BAAI/bge-small-en-v1.5")


def test_seven_agents_integrate_with_mock_model_and_markdown(monkeypatch):
    calls = []

    async def fake_invoke(spec, *, client, settings, narrative, evidence, prior_outputs):
        calls.append(spec.key)
        if spec.key == "investigation":
            return {"incident_summary": "Synthetic runway occupancy conflict.",
                    "documented_facts": [{"text": "Aircraft occupied a runway", "evidence_ids": ["CUR:1"],
                                           "source_ids": [], "evidence_state": "documented_fact"}],
                    "unknowns": ["Clearance sequence requires review"]}
        if spec.key == "risk":
            return {"hazards": [{"hazard_id": "HAZ-01", "description": "Potential conflicting movements",
                                  "evidence_ids": ["CUR:1"]}]}
        if spec.key == "mitigation":
            return {"recommendations": [{"action": "Review clearance coordination checklist",
                                          "hazard_ids": ["HAZ-01"], "priority": "high"}]}
        if spec.key == "critic":
            return {"issues": [], "overall_assessment": "good"}
        if spec.key == "verification":
            return {"claims_needing_source": [], "human_approval_required": False,
                    "completeness_status": "complete"}
        if spec.key == "compliance":
            return {"overall_applicability": "not_assessed"}
        raise AssertionError(spec.key)

    monkeypatch.setattr(service, "invoke_agent", fake_invoke)
    direct = {"document_id": "CUR", "chunk_id": "CUR:1", "source_id": "SRC-SYNTH",
              "source_type": "official_factual", "document_type": "incident_report",
              "evidence_role": "case_factual", "incident_id": "SYNTH-0001", "excerpt": "Synthetic conflict"}
    packet = {key: ([direct] if key != "similar_case" else []) for key in
              ("investigation", "similar_case", "compliance", "risk", "mitigation", "critic", "verification")}
    # Compliance has no authoritative source, so no model call for that role.
    result = asyncio.run(service.analyze_incident(
        "Synthetic runway conflict used only to test the code path.", _settings(),
        case_id="uuid-test", source_incident_id="SYNTH-0001", agent_evidence=packet, retrieval_count=1))
    assert set(result["agent_outputs"]) == set(packet)
    assert calls == ["investigation", "risk", "mitigation", "critic", "verification"]
    assert result["agent_outputs"]["investigation"]["documented_facts"][0]["evidence_ids"] == ["CUR:1"]
    assert result["agent_outputs"]["risk"]["numeric_score"] is None
    assert result["agent_outputs"]["mitigation"]["recommendations"][0]["hazard_ids"] == ["HAZ-01"]
    assert result["agent_outputs"]["verification"]["human_approval_required"] is True
    assert len([t for t in result["trace"] if t.get("step") in packet]) == 7


@pytest.mark.parametrize(("failure", "issue_type"), [
    ("synthetic Critic JSON schema error", "critic_model_output_invalid"),
    ("token_budget: synthetic context over budget", "critic_model_context_budget_exceeded"),
])
def test_critic_model_failure_uses_labeled_deterministic_fallback(monkeypatch, failure, issue_type):
    async def fake_invoke(spec, *, client, settings, narrative, evidence, prior_outputs):
        if spec.key == "critic":
            raise ValueError(failure)
        if spec.key == "investigation":
            return {"incident_summary": "Synthetic runway occupancy conflict."}
        if spec.key == "risk":
            return {"risk_notes": "Needs more evidence."}
        if spec.key == "mitigation":
            return {"recommendations": []}
        if spec.key == "verification":
            return {"claims_needing_source": [], "human_approval_required": True}
        raise AssertionError(spec.key)

    monkeypatch.setattr(service, "invoke_agent", fake_invoke)
    direct = {"document_id": "CUR", "chunk_id": "CUR:1", "source_id": "SRC-SYNTH",
              "source_type": "official_factual", "document_type": "incident_report",
              "evidence_role": "case_factual", "incident_id": "SYNTH-0001", "excerpt": "Synthetic conflict"}
    packet = {key: ([direct] if key not in {"similar_case", "compliance"} else []) for key in
              ("investigation", "similar_case", "compliance", "risk", "mitigation", "critic", "verification")}
    result = asyncio.run(service.analyze_incident(
        "Synthetic runway conflict used only to test Critic fallback.", _settings(),
        case_id="uuid-critic-fallback", source_incident_id="SYNTH-0001",
        agent_evidence=packet, retrieval_count=1,
    ))
    critic = result["agent_outputs"]["critic"]
    assert any(issue["issue_type"] == issue_type for issue in critic["issues"])
    critic_trace = next(event for event in result["trace"] if event["step"] == "critic")
    assert critic_trace["status"] == "deterministic_fallback"


def test_verification_context_budget_error_uses_deterministic_checks(monkeypatch):
    async def fake_invoke(spec, *, client, settings, narrative, evidence, prior_outputs):
        if spec.key == "verification":
            raise ValueError("token_budget: synthetic context over budget")
        if spec.key == "investigation":
            return {"incident_summary": "Synthetic runway occupancy conflict."}
        if spec.key == "risk":
            return {"risk_notes": "Needs more evidence."}
        if spec.key == "mitigation":
            return {"recommendations": []}
        if spec.key == "critic":
            return {"issues": [], "overall_assessment": "requires_human_review"}
        raise AssertionError(spec.key)

    monkeypatch.setattr(service, "invoke_agent", fake_invoke)
    direct = {"document_id": "CUR", "chunk_id": "CUR:1", "source_id": "SRC-SYNTH",
              "source_type": "official_factual", "document_type": "incident_report",
              "evidence_role": "case_factual", "incident_id": "SYNTH-0001", "excerpt": "Synthetic conflict"}
    packet = {key: ([direct] if key not in {"similar_case", "compliance"} else []) for key in
              ("investigation", "similar_case", "compliance", "risk", "mitigation", "critic", "verification")}
    result = asyncio.run(service.analyze_incident(
        "Synthetic runway conflict used only to test verification fallback.", _settings(),
        case_id="uuid-verification-fallback", source_incident_id="SYNTH-0001",
        agent_evidence=packet, retrieval_count=1,
    ))
    verification = result["agent_outputs"]["verification"]
    assert any("Verification input exceeded the safe model context budget" in claim
               for claim in verification["claims_needing_source"])
    assert verification["human_approval_required"] is True
    verification_trace = next(event for event in result["trace"] if event["step"] == "verification")
    assert verification_trace["status"] == "deterministic_fallback"
