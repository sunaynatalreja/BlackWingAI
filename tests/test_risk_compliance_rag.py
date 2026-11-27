"""Risk and Compliance grounding contract tests; no Qwen/IIT/HF requests."""

from blackwing.agents.risk.evidence import ground_risk_output
from blackwing.agents.risk.schema import RiskOutput
from blackwing.agents.compliance.evidence import ground_compliance_output
from blackwing.agents.compliance.schema import ComplianceOutput
from blackwing.rag.agent_router import build_agent_packets
from blackwing.rag.small_context import select_evidence, build_small_messages

FACT = {
    "source_id": "CASE-FACT", "chunk_id": "fact-p2-001", "source_type": "official_factual",
    "document_type": "incident_report", "evidence_role": "case_factual",
    "incident_id": "C-10", "page_number": 2,
    "excerpt": "Aircraft was observed at a runway holding line; clearance unconfirmed.",
}
UPLOAD = {
    "source_id": "USR-1", "evidence_id": "upload-01", "source_type": "user_submitted",
    "document_type": "observer_report", "evidence_role": "reported",
    "incident_id": "C-10", "excerpt": "A witness reported a crossing.",
}
HISTORY = {
    "source_id": "CASE-OTHER", "chunk_id": "history-01", "source_type": "official_final",
    "document_type": "incident_report", "evidence_role": "historical_analogy",
    "incident_id": "OTHER", "excerpt": "A different incident had a readback error.",
}
RULE = {
    "source_id": "RULE-LOCAL", "chunk_id": "rule-2021-p3", "source_type": "operational_order",
    "document_type": "procedure", "jurisdiction": "US", "effective_date": "2021-01-01",
    "section_reference": "2.1", "title": "Runway entry order", "excerpt": "Section 2.1 requires clearance.",
}
FUTURE_RULE = {**RULE, "source_id": "RULE-FUTURE", "chunk_id": "future-p1", "effective_date": "2031-01-01"}
GUIDANCE = {
    "source_id": "SAFE-GUIDE", "chunk_id": "guide-p1", "source_type": "advisory",
    "document_type": "guidance", "jurisdiction": "international",
    "publication_date": "2018-01-01", "excerpt": "Signage and stop bars may reduce runway incursion risk.",
}


def _hazard(evidence_ids):
    return {"hazards": [{"hazard_id": "H1", "description": "Possible runway entry",
             "evidence_ids": evidence_ids, "existing_controls": [], "control_gaps": []}],
            "risk_notes": "Preliminary review", "numeric_score": None, "missing_inputs": []}


def _check(source_id, *, ids=None, section="2.1"):
    return {"checks": [{"source_id": source_id, "section": section,
                         "applicability_status": "violated", "evidence_ids": ids or []}],
            "overall_applicability": "confirmed_noncompliance", "missing_context": []}


def test_risk_keeps_only_supplied_current_case_passage_ids():
    result = ground_risk_output(_hazard(["fact-p2-001", "history-01", "FABRICATED"]),
                                [FACT, HISTORY, GUIDANCE])
    assert RiskOutput.model_validate(result)
    assert result["hazards"][0]["evidence_ids"] == ["fact-p2-001"]
    assert "Removed unsupported" in result["risk_notes"]
    assert result["numeric_score"] is None
    assert result["assessment_status"] == "unassessed_insufficient_inputs"
    assert result["hazards"][0]["risk_index"] is None
    assert result["hazards"][0]["risk_level"] is None


def test_risk_guidance_does_not_establish_incident_fact():
    result = ground_risk_output(_hazard(["guide-p1"]), [GUIDANCE])
    assert result["hazards"][0]["evidence_ids"] == ["guide-p1"]
    assert any("incident-specific" in x for x in result["missing_inputs"])
    assert "hypotheses" in result["risk_notes"]


def test_risk_allows_user_report_but_does_not_score():
    result = ground_risk_output(_hazard(["upload-01"]), [UPLOAD])
    assert result["hazards"][0]["evidence_ids"] == ["upload-01"]
    assert result["numeric_score"] is None


def test_compliance_source_is_real_dated_and_jurisdiction_matching():
    result = ground_compliance_output(_check("RULE-LOCAL", ids=["fact-p2-001", "invented"]),
                                      [RULE, FACT], jurisdiction="US", incident_date="2024-01-01")
    assert ComplianceOutput.model_validate(result)
    assert result["checks"][0]["source_type"] == "operational_order"
    assert result["checks"][0]["applicability_status"] == "requires_human_verification"
    assert result["checks"][0]["evidence_ids"] == ["rule-2021-p3", "fact-p2-001"]
    assert result["overall_applicability"] == "requires_human_verification"


def test_compliance_excludes_future_rule():
    result = ground_compliance_output(_check("RULE-FUTURE"), [FUTURE_RULE],
                                      jurisdiction="US", incident_date="2024-01-01")
    assert result["checks"] == []
    assert result["overall_applicability"] == "not_assessed"


def test_compliance_excludes_other_jurisdiction():
    result = ground_compliance_output(_check("RULE-LOCAL"), [RULE],
                                      jurisdiction="IN", incident_date="2024-01-01")
    assert result["checks"] == []


def test_compliance_requires_incident_date():
    result = ground_compliance_output(_check("RULE-LOCAL"), [RULE],
                                      jurisdiction="US", incident_date=None)
    assert result["checks"] == []


def test_compliance_advisory_never_promoted_to_binding_rule():
    result = ground_compliance_output(_check("SAFE-GUIDE", section=""), [GUIDANCE],
                                      jurisdiction="US", incident_date="2024-01-01")
    assert result["checks"][0]["applicability_status"] == "advisory_only"
    assert result["overall_applicability"] == "advisory_only"


def test_compliance_rejects_invented_clause():
    result = ground_compliance_output(_check("RULE-LOCAL", section="section 999"), [RULE],
                                      jurisdiction="US", incident_date="2024-01-01")
    assert result["checks"][0]["section"] is None
    assert any("exact regulation" in x for x in result["missing_context"])


def test_compliance_unknown_source_cannot_be_cited():
    result = ground_compliance_output(_check("NOT-INDEXED"), [FACT, RULE],
                                      jurisdiction="US", incident_date="2024-01-01")
    assert result["checks"] == []


def test_compliance_compact_evidence_separates_fact_from_rule():
    messages = build_small_messages(
        agent="compliance", system="Check rules only", narrative="Runway holding line reported",
        evidence=[RULE, FACT, HISTORY], prior_outputs={}, upstream_roles=(),
        output_tokens=512, strict_tokens=False,
    )
    import json
    payload = json.loads(messages[1]["content"])
    ids_rule = {x["evidence_id"] for x in payload["case_evidence"]}
    ids_fact = {x["evidence_id"] for x in payload["incident_fact_evidence"]}
    assert "rule-2021-p3" in ids_rule
    assert "fact-p2-001" not in ids_rule
    assert ids_fact == {"fact-p2-001"}
    assert "history-01" not in ids_rule | ids_fact


def test_risk_compact_evidence_excludes_unrelated_case():
    subset = select_evidence("risk", [FACT, GUIDANCE, HISTORY], limit=3)
    assert {x["evidence_id"] for x in subset} == {"fact-p2-001", "guide-p1"}


class _Retriever:
    def search(self, query, documents, *, scope, **kwargs):
        if scope == "direct":
            return {"results": [FACT] if kwargs.get("case_id") == "C-10" else []}
        if scope == "regulatory":
            return {"results": [RULE] if kwargs.get("jurisdiction") == "US" and kwargs.get("incident_date") else []}
        if scope == "guidance":
            return {"results": [GUIDANCE]}
        return {"results": [HISTORY]}


def test_rag_agent_packet_feeds_rule_fact_and_guidance_in_three_items():
    packets = build_agent_packets(retriever=_Retriever(), narrative="Runway crossing reported",
                                  case_id="C-10", jurisdiction="US", incident_date="2024-01-01",
                                  documents=[], limit_per_agent=3)
    compliance = [x["source_id"] for x in packets["compliance"]]
    risk = [x["source_id"] for x in packets["risk"]]
    assert compliance == ["RULE-LOCAL", "CASE-FACT", "SAFE-GUIDE"]
    assert "CASE-FACT" in risk and "SAFE-GUIDE" in risk
    assert "CASE-OTHER" not in risk + compliance


def test_agent_workflow_postprocesses_rag_citations(monkeypatch):
    """Risk/Compliance hooks execute at the existing LangGraph node boundary."""
    from types import SimpleNamespace
    from blackwing import service
    from blackwing.agents.registry import AGENT_BY_ID

    async def mock_invoke(spec, *, client, settings, narrative, evidence, prior_outputs):
        if spec.key == "risk":
            assert "CASE-FACT" in {x["source_id"] for x in evidence}
            return _hazard(["fact-p2-001", "history-01"])
        if spec.key == "compliance":
            assert "RULE-LOCAL" in {x["source_id"] for x in evidence}
            assert "CASE-FACT" in {x["source_id"] for x in evidence}
            return _check("RULE-LOCAL", ids=["fact-p2-001", "FAKE"])
        raise AssertionError("Only two roles should be invoked in this test")

    monkeypatch.setattr(service, "invoke_agent", mock_invoke)
    settings = SimpleNamespace(model_name="mock-qwen-no-network")
    state = {
        "narrative": "An aircraft approached the holding position.",
        "jurisdiction": "US", "incident_date": "2024-01-01",
        "evidence": [], "agent_evidence": {
            "risk": [FACT, GUIDANCE], "compliance": [RULE, FACT, GUIDANCE]
        },
        "outputs": {}, "trace": [],
    }
    import asyncio
    compliance_result = asyncio.run(service._agent_node(AGENT_BY_ID["compliance"], None, settings)(state))
    state.update(compliance_result)
    risk_result = asyncio.run(service._agent_node(AGENT_BY_ID["risk"], None, settings)(state))
    compliance = compliance_result["outputs"]["compliance"]
    risk = risk_result["outputs"]["risk"]
    assert compliance["checks"][0]["applicability_status"] == "requires_human_verification"
    assert compliance["checks"][0]["evidence_ids"] == ["rule-2021-p3", "fact-p2-001"]
    assert risk["hazards"][0]["evidence_ids"] == ["fact-p2-001"]
    assert risk["numeric_score"] is None
