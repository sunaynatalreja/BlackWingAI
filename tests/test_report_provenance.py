"""Report handoff and provenance regressions; all model IO is mocked."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from blackwing import service
from blackwing.agents.report.agent import SPEC
from blackwing.agents.report.provenance import collect_report_evidence, ground_report_point
from blackwing.agents.report.schema import ReportSectionDraft, normalize_report_section
from blackwing.agents.report.workflow import _ground_sections, _merge_source_detail, create_report
from blackwing.agents.runtime import invoke_agent
from blackwing.agents.base import AgentSpec
from blackwing.report_quality import assess_quality
from test_agentic_local import settings
from test_report_quality_v2 import sample_report


OFFICIAL = {'chunk_id': 'FACT:1', 'source_id': 'FACT', 'source_type': 'official_factual',
            'evidence_role': 'case_factual', 'incident_id': 'CASE', 'excerpt': 'Aircraft crossed a holding line.'}
ACCOUNT = {'chunk_id': 'USER:1', 'source_id': 'USER', 'source_type': 'user_submitted',
           'evidence_role': 'reported', 'incident_id': 'CASE', 'excerpt': 'A technician reports an illuminated sign.'}


def point(ref='USER:1', classification='documented_fact'):
    return {'text': 'A technician reports an illuminated sign.', 'classification': classification,
            'evidence_ids': [ref], 'source_ids': []}


@pytest.mark.parametrize('row,incident,expected', [
    (ACCOUNT, 'CASE', 'reported_observation'),
    (OFFICIAL, 'CASE', 'documented_fact'),
    (OFFICIAL, 'OTHER', 'verification_gap'),
    ({**OFFICIAL, 'evidence_role': 'historical_analogy'}, 'CASE', 'historical_analogy'),
    ({**OFFICIAL, 'evidence_status': 'UNVERIFIED'}, 'CASE', 'reported_observation'),
])
def test_source_provenance_caps_model_fact_labels(row, incident, expected):
    p = point(row['chunk_id'])
    original = deepcopy((p, row))
    assert ground_report_point(p, [row], incident_id=incident)['classification'] == expected
    assert (p, row) == original


def test_source_id_or_phantom_agent_reference_is_not_a_factual_passage():
    points = [point('PHANTOM'), {**point(), 'evidence_ids': [], 'source_ids': ['FACT']}]
    result = _ground_sections([{'key_points': points}], {
        'agent_outputs': {'fabricated': {'evidence_ids': ['PHANTOM']}}, 'retrieved_evidence': [OFFICIAL]})
    assert all(p['classification'] == 'verification_gap' for p in result[0]['key_points'])


def test_deduplication_cannot_upgrade_a_validated_attributed_account():
    prior = {'investigation': {'reported_observations': [
        {'text': point()['text'], 'evidence_state': 'reported_observation', 'evidence_ids': ['USER:1']}]}}
    original = deepcopy(prior)
    section = _merge_source_detail({'summary': 'Summary', 'key_points': [point()]}, 'overview', prior)
    assert len(section['key_points']) == 1
    assert section['key_points'][0]['classification'] == 'reported_observation'
    assert prior == original


def test_report_packet_retains_agent_evidence_without_overwriting_conflicting_provenance():
    rows = collect_report_evidence([ACCOUNT], {'investigation': [OFFICIAL],
        'risk': [OFFICIAL, {**OFFICIAL, 'evidence_role': 'historical_analogy'}]})
    assert len(rows) == 3
    rows[0]['excerpt'] = 'changed copy'
    assert ACCOUNT['excerpt'] != 'changed copy'
    assert ground_report_point(point('FACT:1'), rows, incident_id='CASE')['classification'] == 'verification_gap'


def test_deterministic_report_preserves_accounts_conflicts_and_critic_without_invented_testimony(monkeypatch):
    from blackwing.agents.report import workflow
    async def forbidden(*args, **kwargs):
        raise AssertionError('A deterministic report must not call a model')
    monkeypatch.setattr(workflow, 'invoke_agent', forbidden)
    prior = {'investigation': {'incident_summary': 'Unverified account',
                              'unknowns': ['Vehicle position is unknown.']},
             'critic': {'issues': [{'description': 'Clearance sequence remains unresolved.'}]},
             'verification': {'conflicts': ['Two records identify different aircraft.']}}
    async def run():
        async with httpx.AsyncClient() as client:
            return await create_report(client=client, settings=settings(), narrative='VERIFICATION FOCUS\nCheck radio recording.',
                evidence=[ACCOUNT], prior_outputs=prior, deterministic=True, incident_id='CASE')
    report, trace = asyncio.run(run())
    visible = json.dumps(report)
    assert ACCOUNT['excerpt'] in visible
    assert 'Vehicle position is unknown.' in visible
    assert 'Clearance sequence remains unresolved.' in visible
    assert 'Two records identify different aircraft.' in visible
    assert 'Check radio recording.' in visible
    assert 'driver reported stopping' not in visible.casefold()
    assert not any(p['classification'] == 'documented_fact' for s in report['sections'] for p in s['key_points'])
    assert trace[0]['status'] == 'deterministic_fallback'


def test_real_report_node_receives_the_agents_factual_packet():
    state = {'case_id': 'internal-case', 'source_incident_id': 'CASE', 'narrative': 'Reported crossing',
             'evidence': [ACCOUNT], 'agent_evidence': {'investigation': [OFFICIAL]},
             'plan': {'task_modes': {'report': 'deterministic_assembly'}},
             'outputs': {'investigation': {'incident_summary': 'Reported crossing', 'documented_facts': [
                 {'text': OFFICIAL['excerpt'], 'evidence_state': 'documented_fact', 'evidence_ids': ['FACT:1']}]}}}
    async def run():
        async with httpx.AsyncClient() as client:
            return await service._agent_node(SPEC, client, settings())(state)
    result = asyncio.run(run())
    points = result['outputs']['report']['sections'][0]['key_points']
    assert any(p['classification'] == 'documented_fact' and p['evidence_ids'] == ['FACT:1'] for p in points)
    assert any(p['classification'] == 'reported_observation' and p['evidence_ids'] == ['USER:1'] for p in points)


def test_runtime_normalizes_report_shape_before_schema_validation():
    from blackwing.agents.report.workflow import _PACKAGE
    payload = {'section_summary': 'Attributed account', 'findings': [
        {'claim': 'Witness reports a crossing.', 'type': 'reported', 'citations': 'USER:1'}],
        'task_id': 'risk_and_follow_up'}
    def reply(request):
        return httpx.Response(200, json={'message': {'content': json.dumps(payload)}, 'done_reason': 'stop'})
    spec = AgentSpec(key='report_overview', name='Overview', prompt_path=_PACKAGE/'overview.md',
                     output_schema=ReportSectionDraft, upstream_context=())
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
            return await invoke_agent(spec, client=client, settings=settings(), narrative='A submitted account',
                                      evidence=[ACCOUNT], prior_outputs={})
    result = asyncio.run(run())
    assert result['summary'] == 'Attributed account'
    assert result['key_points'][0]['classification'] == 'reported_observation'
    assert 'task_id' not in result


def test_empty_normalized_model_section_is_not_a_success():
    with pytest.raises(ValueError):
        normalize_report_section({'summary': ' ', 'findings': [{}]})


def test_completeness_credits_attribution_without_certifying_truth():
    report = sample_report()
    inv = report['agent_outputs']['investigation']
    inv['reported_observations'] = [{**inv['documented_facts'][0], 'evidence_state': 'reported_observation'}]
    inv['documented_facts'] = []
    report['generated_report']['sections'][0]['key_points'][0]['classification'] = 'reported_observation'
    result = assess_quality({}, report)
    assert result['rubric_version'] == '2.1'
    assert sum(c['weight'] for c in result['checks']) == 100
    assert next(c for c in result['checks'] if c['check'] == 'Current-case evidence represented')['earned'] == 12
    assert any('not independently verified' in f for f in result['flags'])


def test_empty_claims_and_citation_only_entries_do_not_earn_coverage():
    report = sample_report()
    report['agent_outputs']['investigation']['documented_facts'] = [{'evidence_ids': ['USER:1']}]
    report['agent_outputs']['investigation']['reported_observations'] = [{}]
    report['generated_report']['sections'][0]['key_points'] = [
        {'text': ' ', 'classification': 'documented_fact', 'evidence_ids': ['USER:1']}]
    result = assess_quality({}, report)
    for check in result['checks']:
        if check['check'] in {'Current-case evidence represented', 'Factual evidence references present', 'Report factual points referenced'}:
            assert check['earned'] == 0
