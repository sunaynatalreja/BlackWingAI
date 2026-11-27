from blackwing.agents.investigation.schema import InvestigationOutput
from blackwing.agents.report.schema import REPORT_TASK_ORDER, ReportPlanOutput, normalize_report_plan_payload
from blackwing.agents.shared_schema import normalize_evidence_state_labels


def test_unrecognized_evidence_state_is_conservatively_downgraded_to_unknown():
    output = InvestigationOutput.model_validate({
        'incident_summary': 'A factor was proposed.',
        'contributing_factors': [{
            'text': 'Possible coordination issue.',
            'evidence_state': 'preliminary_assessment',
            'evidence_ids': ['EV-1'],
        }],
    })
    assert output.contributing_factors[0].evidence_state == 'unknown'


def test_runtime_evidence_state_normalizer_reports_changes_without_touching_claims():
    payload = {'contributing_factors': [{'text': 'Claim', 'evidence_state': 'unsupported_label'}]}
    changed = normalize_evidence_state_labels(payload)
    assert changed == 1
    assert payload['contributing_factors'][0] == {
        'text': 'Claim', 'evidence_state': 'unknown'}


def test_report_planner_task_order_alias_normalizes_to_fixed_schema_order():
    payload, changed = normalize_report_plan_payload({'task_order': list(reversed(REPORT_TASK_ORDER))})
    assert changed is True
    assert payload == {'task_ids': list(REPORT_TASK_ORDER)}
    assert ReportPlanOutput.model_validate(payload).task_ids == list(REPORT_TASK_ORDER)
