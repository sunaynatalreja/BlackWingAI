from blackwing.agents.evidence_utils import is_direct
from blackwing.agents.risk.evidence import ground_risk_output


def test_other_incident_cannot_be_promoted_to_current_case():
    row = {"chunk_id": "BW-EV-004:1", "source_type": "official_preliminary",
           "evidence_role": "case_factual", "incident_id": "IND-AAIB-003"}
    assert not is_direct(row, "IND-AAIB-002")
    assert not is_direct({**row, "incident_id": ""}, "IND-AAIB-002")
    assert is_direct(row, "IND-AAIB-003")


def test_other_incident_cannot_support_risk_rating():
    row = {"chunk_id": "BW-EV-004:1", "source_type": "official_preliminary",
           "evidence_role": "case_factual", "incident_id": "IND-AAIB-003"}
    output = {"hazards": [{"hazard_id": "H1", "description": "Potential conflict",
                           "evidence_ids": ["BW-EV-004:1"], "severity": 3,
                           "likelihood": "C", "severity_basis": "Possible collision",
                           "likelihood_basis": "Unclear communications"}]}
    hazard = ground_risk_output(output, [row], incident_id="IND-AAIB-002")["hazards"][0]
    assert hazard["risk_index"] is None
    assert hazard["severity"] is None
    assert hazard["likelihood"] is None


def test_single_current_incident_does_not_establish_likelihood():
    row = {"chunk_id": "FACT:1", "source_type": "official_factual",
           "evidence_role": "case_factual", "incident_id": "CURRENT"}
    output = {"hazards": [{"hazard_id": "H1", "description": "Potential conflict",
                           "evidence_ids": ["FACT:1"], "severity": 3,
                           "likelihood": "C", "severity_basis": "Possible collision",
                           "likelihood_basis": "Unclear communications"}]}
    result = ground_risk_output(output, [row], incident_id="CURRENT")
    assert result["hazards"][0]["risk_index"] is None
    assert result["assessment_status"] == "unassessed_insufficient_inputs"
