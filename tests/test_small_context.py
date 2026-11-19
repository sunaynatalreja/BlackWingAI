from __future__ import annotations

import pytest

from blackwing.rag.small_context import build_small_messages, select_evidence, verify_retry_budget


DOCS = [
    {"source_id": "CASE-OBS", "evidence_id": "OBS-1", "source_type": "user_submitted", "content": "Vehicle reported near runway 27."},
    {"source_id": "LAW-1", "chunk_id": "LAW-1:p1", "source_type": "regulation", "jurisdiction": "IN", "effective_date": "2024-01-01", "excerpt": "Runway procedure text."},
    {"source_id": "ADVICE-1", "chunk_id": "ADVICE-1:p3", "source_type": "advisory", "excerpt": "Surface movement monitoring."},
    {"source_id": "PAST-1", "chunk_id": "PAST-1:p4", "source_type": "official_final", "evidence_role": "historical_analogy", "excerpt": "Probable cause in an unrelated event."},
]


def test_investigation_excludes_regulations_and_historical_analogy():
    a = select_evidence("investigation", DOCS)
    assert [x["source_id"] for x in a] == ["CASE-OBS"]


def test_compliance_only_rules_and_advisory():
    ids = [x["source_id"] for x in select_evidence("compliance", DOCS)]
    assert ids == ["LAW-1", "ADVICE-1"]


def test_risk_no_historical_as_current_fact():
    ids = [x["source_id"] for x in select_evidence("risk", DOCS)]
    assert "PAST-1" not in ids
    assert "CASE-OBS" in ids


def test_citations_remain_true_ids():
    x = select_evidence("compliance", DOCS)[0]
    assert x["source_id"] == "LAW-1" and x["evidence_id"] == "LAW-1:p1"
    assert x["effective_date"] == "2024-01-01"


def test_compact_output_remains_bounded_and_warns_truncation():
    messages = build_small_messages(agent="risk", system="Generate JSON", narrative="X" * 4000,
                                    evidence=DOCS, prior_outputs={}, upstream_roles=(), output_tokens=512)
    assert "TRUNCATED" in messages[1]["content"]
    assert "PAST-1" not in messages[1]["content"]


def test_large_prior_output_is_capped_and_fits():
    messages = build_small_messages(agent="critic", system="Generate JSON", narrative="Short fact.",
                                    evidence=DOCS, prior_outputs={"investigation": {"summary": "Y" * 100000}},
                                    upstream_roles=("investigation",), output_tokens=768)
    assert len(messages[1]["content"]) < 5000


def test_final_review_context_summarizes_agent_outputs_to_fit():
    long_claim = {"text": "Incident claim " + "X" * 800, "evidence_ids": ["E" * 200] * 8,
                  "source_ids": ["S" * 200] * 8, "evidence_state": "unknown"}
    outputs = {
        "investigation": {
            "incident_summary": "Summary " + "I" * 3000,
            "timeline_events": [{"event": "Event " + "T" * 500, "time_reference": "unknown",
                                 "evidence_ids": ["E" * 200] * 8, "evidence_state": "unknown"} for _ in range(8)],
            **{key: [long_claim] * 8 for key in
               ("documented_facts", "reported_observations", "hypotheses", "contributing_factors")},
            "unknowns": ["Unknown " + "U" * 500] * 8,
        },
        "similar_case": {"case_status": "unseen_case", "matches": [{"source_id": "NTSB",
            "title": "Historical case " + "H" * 500, "similarity": "Similar " + "S" * 500,
            "differences_or_unknowns": ["Difference " + "D" * 500] * 8} for _ in range(8)]},
        "compliance": {"jurisdiction": "US", "overall_applicability": "not_assessed",
            "checks": [{"source_id": "RULE", "source_type": "regulation", "effective_date": "unknown",
                        "applicability_status": "requires_review", "rationale": "R" * 500,
                        "evidence_ids": ["E" * 200] * 8} for _ in range(8)],
            "missing_context": ["Missing " + "M" * 500] * 8},
        "risk": {"hazards": [{"hazard_id": "H001", "description": "Hazard " + "Z" * 500,
            "evidence_ids": ["E" * 200] * 8, "existing_controls": ["C" * 300] * 8,
            "control_gaps": ["G" * 300] * 8} for _ in range(8)], "risk_notes": "N" * 1000,
            "missing_inputs": ["Input " + "M" * 500] * 8},
        "mitigation": {"recommendations": [{"hazard_ids": ["H001"] * 8,
            "contributing_factor": "Factor " + "F" * 500, "action": "Action " + "A" * 500,
            "classification": "preventive", "priority": "proposed"} for _ in range(8)],
            "assumptions": ["Assumption " + "A" * 500] * 8},
        "critic": {"issues": [{"issue_type": "gap", "severity": "review", "affected_output": "risk",
            "description": "Issue " + "I" * 500, "suggested_correction": "Correction " + "C" * 500} for _ in range(8)],
            "missing_analysis": ["Missing " + "M" * 500] * 8},
    }
    for agent, roles in (
        ("critic", ("investigation", "similar_case", "compliance", "risk", "mitigation")),
        ("verification", ("investigation", "similar_case", "compliance", "risk", "mitigation", "critic")),
    ):
        messages = build_small_messages(agent=agent, system="Review outputs for gaps. " * 30,
            narrative="Current incident. " * 100, evidence=DOCS, prior_outputs=outputs,
            upstream_roles=roles, output_tokens=1024)
        assert "Incident claim" in messages[1]["content"]
        assert len(messages[1]["content"]) < 12000


def test_invalid_output_ceiling_fails_closed():
    with pytest.raises(ValueError, match="1024"):
        build_small_messages(agent="risk", system="X", narrative="Y", evidence=[],
                             prior_outputs={}, upstream_roles=(), output_tokens=1025)


def test_oversized_static_system_prompt_fails():
    with pytest.raises(ValueError, match="token_budget"):
        build_small_messages(agent="risk", system="A" * 90000, narrative="Test", evidence=[],
                             prior_outputs={}, upstream_roles=(), output_tokens=512)


def test_retry_budget_bounded():
    with pytest.raises(ValueError, match="token_budget"):
        verify_retry_budget([{"role": "user", "content": "P" * 90000}], 512, strict=False)


def test_strict_hosted_preflight_checks_exact_count(monkeypatch):
    from blackwing.rag import small_context

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is True and add_generation_prompt is True
            return list(range(230))

    monkeypatch.setattr(small_context, "_tokenizer", lambda model: FakeTokenizer())
    msgs = build_small_messages(agent="compliance", system="Required policy", narrative="Runway 12",
                                evidence=DOCS, prior_outputs={}, upstream_roles=(), output_tokens=512,
                                strict_tokens=True)
    assert len(msgs) == 2


def test_strict_hosted_fails_if_tokenizer_unavailable(monkeypatch):
    from blackwing.rag import small_context

    def unavailable(_):
        raise OSError("No cached tokenizer")

    monkeypatch.setattr(small_context, "_tokenizer", unavailable)
    with pytest.raises(ValueError, match="tokenizer"):
        build_small_messages(agent="risk", system="Policy", narrative="Runway 12",
                             evidence=DOCS, prior_outputs={}, upstream_roles=(), output_tokens=512,
                             strict_tokens=True)


def test_agent_runtime_uses_bounded_per_role_evidence(tmp_path):
    """Actual agent invocation path; mocked HTTP, no model downloads or tokens."""
    import asyncio
    import json
    import httpx
    from pydantic import BaseModel
    from blackwing.agents.base import AgentSpec
    from blackwing.agents.runtime import invoke_agent
    from blackwing.config import Settings

    class TinyOutput(BaseModel):
        assessment: str

    prompt = tmp_path / "prompt.md"
    prompt.write_text("Return JSON report", encoding="utf-8")
    spec = AgentSpec("risk", "Risk", prompt, TinyOutput, ("investigation",))
    requests = []

    def response(req):
        sent = json.loads(req.content)
        requests.append(sent)
        assert sent["options"]["num_predict"] == 512
        assert "PAST-1" not in sent["messages"][1]["content"]
        assert "CASE-OBS" in sent["messages"][1]["content"]
        return httpx.Response(200, json={"message": {"content": '{"assessment":"Provisional"}'}})

    settings = Settings(
        host="127.0.0.1", port=8000, llm_provider="ollama", ollama_base_url="http://localhost:11434",
        ollama_model="qwen3:8b", ollama_timeout_seconds=15, ollama_temperature=0,
        ollama_think=False, ollama_max_output_tokens=512, api_base_url="", api_model="",
        api_key="", api_timeout_seconds=15, api_temperature=0, api_max_output_tokens=512,
        ui_api_url="", ui_request_timeout_seconds=15, rag_embedding_model="BAAI/bge-small-en-v1.5"
    )

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            return await invoke_agent(spec, client=client, settings=settings, narrative="A vehicle entered runway 27.",
                                      evidence=DOCS, prior_outputs={"investigation": {"summary": "reported only"}})

    outcome = asyncio.run(go())
    assert outcome["assessment"] == "Provisional"
    assert len(requests) == 1
