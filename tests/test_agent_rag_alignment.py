from __future__ import annotations

from blackwing.rag.agent_router import AGENT_ROLES, build_agent_packets, packet_diagnostics
from blackwing.rag.engine import HybridIndex
from blackwing.rag.tokenization import ApproximateTokenizer


def _docs():
    return [
        dict(document_id="D-CURRENT", source_id="FACT-A", incident_id="CASE-ALPHA",
             incident_date="2024-03-05", source_type="official_factual", evidence_role="case_factual",
             document_type="incident_report", jurisdiction="US", page_number=2,
             title="Current event factual note", text="Aircraft entered runway holding position after taxiway turn and clearance inquiry."),
        dict(document_id="D-OTHER", source_id="FACT-B", incident_id="CASE-BETA",
             incident_date="2022-04-01", source_type="official_final", evidence_role="historical_analogy",
             document_type="incident_report", jurisdiction="US", page_number=4,
             title="Historical runway incursion", text="Aircraft runway occupancy conflict was reported after ATC clearance readback."),
        dict(document_id="D-RULE", source_id="RULE-2020", source_type="operational_order",
             document_type="procedure", jurisdiction="US", effective_date="2020-01-01",
             publication_date="2019-12-20", page_number=11, title="Movement area procedures",
             text="Runway holding position clearance and mandatory readback procedures for ground aircraft."),
        dict(document_id="D-FUTURE", source_id="RULE-2030", source_type="operational_order",
             document_type="procedure", jurisdiction="US", effective_date="2030-01-01",
             publication_date="2030-01-01", page_number=3, title="Future procedures",
             text="New runway holding position clearance requirement issued 2030."),
        dict(document_id="D-GUIDE", source_id="GUIDE-2018", source_type="advisory",
             evidence_role="general_guidance", document_type="guidance", publication_date="2018-02-01",
             jurisdiction="US", title="Runway safety guidance", page_number=5,
             text="Runway incursion hazards: signage, stop bars, situational awareness and safety barriers."),
        dict(document_id="D-GUIDE-FUTURE", source_id="GUIDE-2027", source_type="advisory",
             evidence_role="general_guidance", document_type="guidance", publication_date="2027-01-01",
             jurisdiction="US", title="Future runway safety recommendations", page_number=8,
             text="Future runway incursion risk mitigation guidance about safety barriers."),
    ]


class LexicalRetriever:
    """Uses the team's *real* BM25 ranker and evidence filters, without fake metrics."""
    def __init__(self):
        self.index = HybridIndex.from_documents(_docs(), strategy="fixed", tokenizer=ApproximateTokenizer())

    def search(self, query, documents, *, limit, method, scope, **kwargs):
        return self.index.search(query, top_k=limit, method="bm25", scope=scope,
                                 case_id=kwargs.get("case_id"), jurisdiction=kwargs.get("jurisdiction"),
                                 incident_date=kwargs.get("incident_date"))


def _packets(case_id="CASE-ALPHA", jurisdiction="US", date="2024-03-05", uploads=None):
    return build_agent_packets(
        retriever=LexicalRetriever(),
        narrative="Aircraft entered runway holding position after taxiway turn without verified ATC clearance",
        case_id=case_id, jurisdiction=jurisdiction, incident_date=date,
        documents=[], uploaded_evidence=uploads or [], method="bm25",
    )


def test_has_exactly_sunayanas_seven_roles():
    result = _packets()
    assert list(result) == list(AGENT_ROLES)
    assert packet_diagnostics(result)["all_seven_roles_present"]
    assert all(len(evidence) <= 3 for evidence in result.values())


def test_current_case_not_confused_with_other_incident():
    packets = _packets()
    assert "FACT-A" in {r["source_id"] for r in packets["investigation"]}
    assert "FACT-B" not in {r["source_id"] for r in packets["investigation"]}
    assert "FACT-B" not in {r["source_id"] for r in packets["risk"]}
    assert "FACT-B" not in {r["source_id"] for r in packets["verification"]}
    assert "FACT-A" not in {r["source_id"] for r in packets["similar_case"]}


def test_historical_comparison_labeled_analogy():
    similar = _packets()["similar_case"]
    assert similar
    assert all(item["evidence_role"] == "historical_analogy" for item in similar)
    assert all("OTHER INCIDENT" in item["caution"] for item in similar)


def test_compliance_only_prior_valid_rules_and_nonbinding_guidance():
    sources = {r["source_id"] for r in _packets()["compliance"]}
    assert "RULE-2020" in sources
    assert "RULE-2030" not in sources
    assert "GUIDE-2027" not in sources
    assert "GUIDE-2018" in sources


def test_without_date_or_jurisdiction_cannot_suggest_binding_rules():
    assert not any(r["source_type"] == "operational_order"
                   for r in _packets(date=None)["compliance"])
    assert not any(r["source_type"] == "operational_order"
                   for r in _packets(jurisdiction=None)["compliance"])


def test_risk_and_mitigation_get_guidance_not_historical_findings():
    p = _packets()
    assert "GUIDE-2018" in {r["source_id"] for r in p["risk"]}
    assert "GUIDE-2018" in {r["source_id"] for r in p["mitigation"]}
    assert all(r["source_id"] != "FACT-B" for r in [*p["risk"], *p["mitigation"]])


def test_new_case_without_matching_direct_record_cannot_borrow_foreign_facts():
    p = _packets(case_id="BRAND-NEW-INCIDENT")
    assert not p["investigation"]
    assert "FACT-B" in {r["source_id"] for r in p["similar_case"]}


def test_user_uploaded_evidence_is_reported_not_official():
    p = _packets(case_id="BRAND-NEW-INCIDENT", uploads=[{
        "evidence_id": "UPLOAD-001", "excerpt": "Observer saw vehicle near runway 16",
        "document_type": "observer_report", "title": "Unverified observer note",
    }])
    inv = p["investigation"]
    assert len(inv) == 1
    assert inv[0]["source_type"] == "user_submitted"
    assert inv[0]["evidence_status"] == "USER_REPORTED_UNVERIFIED"
    assert inv[0]["incident_id"] == "BRAND-NEW-INCIDENT"


def test_role_packet_limits_and_bad_case_id():
    try:
        _packets(case_id="")
        raise AssertionError("Invalid empty case identity accepted")
    except ValueError:
        pass
    try:
        build_agent_packets(retriever=LexicalRetriever(), narrative="x", case_id="CASE-ALPHA",
                            jurisdiction="US", incident_date="2024-03-05", documents=[], limit_per_agent=10)
        raise AssertionError("Too large packet limit accepted")
    except ValueError:
        pass
