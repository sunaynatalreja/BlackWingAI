"""Focused synthetic-report fixes. Providers and policy references are mocked."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest

from blackwing import synthetic_gold as policy
from blackwing.case_inputs import attributed_synthetic_summary
from blackwing.agents.report.workflow import create_report
from blackwing.agents.critic.grounding import audit_critic_output
from test_agentic_local import settings

BIRCH = ('CASE ID: BW-UNSEEN-002\nORIGINAL FICTIONAL training scenario at Birch Ferry Airport. '
         'A training crew reports receiving a taxi-route amendment while a cabin call interrupted the readback. '
         'The controller note records an acknowledgement but does not identify the words used.')
CORAL = ('ORIGINAL FICTIONAL training scenario at Coral Basin Aerodrome. '
         'An operations vehicle was tasked with checking a runway light. A radio note says the inspection ended, '
         'while a camera screenshot cannot place the vehicle relative to the protected area. '
         'No event date or country/jurisdiction is supplied. These are unverified synthetic accounts, not official findings.')
TEXT = 'Regulatory applicability and risk remain unassessed; no proven cause is established; historical matches are candidate analogies only; recommendations are deferred pending evidence.'
REPORT = {'generated_report':{'executive_summary':TEXT, 'sections':[]},
          'agent_outputs':{'investigation':{'incident_summary':'Hidden model answer is not visible report text.'}}}
ANSWER = 'Retain uncertainty and require verification before making a factual determination.'


def refs():
    return ({key:{'question_id':key, 'query':label[0], 'approved gold answer':ANSWER}
             for key,label in policy.CRITERIA.items()}, 'mock-reference-hash')


def verdict(status='supported'):
    return {'status':status, 'report_field':'executive_summary', 'report_quote':TEXT,
            'gold_quote':ANSWER, 'rationale':'The supplied report explicitly preserves uncertainty.'}


@pytest.mark.parametrize('narrative,wrong_summary', [
    (BIRCH, 'The crew did not fully read or understand the amendment.'),
    (CORAL, 'The vehicle completed the inspection inside the protected area.')])
def test_report_assembly_preserves_supplied_wording_without_an_extra_model_call(monkeypatch, narrative, wrong_summary):
    from blackwing.agents.report import workflow
    async def forbidden(*args, **kwargs):
        raise AssertionError('Deterministic report formatting must not call a model.')
    monkeypatch.setattr(workflow, 'invoke_agent', forbidden)
    prior = {'investigation':{'incident_summary':wrong_summary, 'unknowns':['Clearance and position remain unverified.']}}
    original = deepcopy(prior)
    async def build():
        async with httpx.AsyncClient() as client:
            return await create_report(client=client, settings=settings(), narrative=narrative,
                                       evidence=[], prior_outputs=prior, deterministic=True)
    generated, _ = asyncio.run(build())
    assert generated['executive_summary'] == attributed_synthetic_summary(narrative)
    assert generated['executive_summary'] == generated['sections'][0]['summary']
    assert wrong_summary not in generated['executive_summary'] and prior == original
    assert 'Unverified fictional scenario' in generated['executive_summary']
    assert attributed_synthetic_summary('Investigate JFK DCA23LA125.') is None


def test_critic_accepts_disclosed_limits_and_deferral_but_retains_actual_model_issues():
    prior = {'investigation':{'documented_facts':[{'text':'An account only.', 'evidence_state':'reported_observation',
               'evidence_ids':['CASE:1']}], 'unknowns':['Vehicle position is unverified.']},
             'risk':{'hazards':[], 'assessment_status':'unassessed_insufficient_inputs'},
             'compliance':{'overall_applicability':'not_assessed', 'missing_context':['Date and jurisdiction unknown.']},
             'mitigation':{'recommendations':[], 'assumptions':['Deferred pending verified position evidence.']}}
    real_issue = {'issue_type':'summary_overstatement', 'severity':'review', 'affected_output':'investigation',
                  'description':'The summary asserts a completed inspection without support.', 'suggested_correction':None}
    output = audit_critic_output({'issues':[real_issue]}, prior)
    assert output['issues'] == [real_issue]
    assert output['overall_assessment'] == 'requires_human_review'


def test_policy_packet_contains_visible_report_only_and_enforces_exact_meaningful_quotes():
    reference = refs()[0]['BW-G061']
    packet = policy._view('BW-G061', {}, REPORT)
    assert packet['report_fields']['executive_summary'] == TEXT
    assert 'Hidden model answer' not in json.dumps(packet)
    assert policy.validate_judgment(verdict(), reference, {}, REPORT)['status'] == 'supported'
    for change in ({'report_field':'report.agent_outputs.risk.assessment_status'},
                   {'report_quote':'[]'}, {'report_quote':'Fabricated evidence quotation'},
                   {'gold_quote':'Fabricated approved answer'}, {'status':'not_applicable'}):
        with pytest.raises(ValueError):
            policy.validate_judgment({**verdict(), **change}, reference, {}, REPORT)
    assert policy.validate_judgment(verdict('not_applicable'), refs()[0]['BW-G078'], {}, REPORT)['status'] == 'not_applicable'


def test_default_judge_adapter_resolves_selected_excerpt_without_model_copied_quotes(monkeypatch):
    from blackwing import gold_comparison
    async def mock_call(config, system, payload, validator):
        assert 'report_excerpts' in payload and 'policy_question' not in payload
        assert 'withhold binding compliance' in payload['safety_check']
        assert 'Do not answer an incident question' in system
        return validator({'status':'supported', 'excerpt_id':next(iter(payload['report_excerpts'])),
                          'rationale':'The report explicitly withholds unsupported findings.'})
    monkeypatch.setattr(gold_comparison, 'validated_evaluation', mock_call)
    result = asyncio.run(policy.judge_policy(settings(), 'BW-G061', refs()[0]['BW-G061'], {}, REPORT))
    assert policy.validate_judgment(result, refs()[0]['BW-G061'], {}, REPORT)['status'] == 'supported'
    assert result['report_quote'] in TEXT and result['gold_quote'] == ANSWER


def test_excerpt_resolution_rejects_invented_ids_and_cannot_force_a_passing_status():
    reference=refs()[0]['BW-G061']
    excerpts=policy._excerpts(REPORT)
    decision={'status':'supported','excerpt_id':'FAKE','rationale':'Unsupported selector.'}
    with pytest.raises(ValueError):
        policy.resolve_decision(decision,reference,{},REPORT,excerpts)
    excluded={**decision,'status':'not_applicable','excerpt_id':next(iter(excerpts))}
    with pytest.raises(ValueError):
        policy.resolve_decision(excluded,reference,{},REPORT,excerpts)
    unknown=policy.resolve_decision({**decision,'status':'not_verified','excerpt_id':None},reference,{},REPORT,excerpts)
    assert unknown['status']=='not_verified' and not unknown['report_quote']


def test_all_visible_excerpt_text_including_short_claims_remains_available_for_review():
    report={'generated_report':{'executive_summary':('A long attributed observation. '*30)+'Pilot did.'}}
    excerpts=policy._excerpts(report)
    assert any(row['text']=='Pilot did.' for row in excerpts.values())
    assert all(len(row['text'])<=240 and row['text'] in report['generated_report']['executive_summary']
               for row in excerpts.values())


def test_policy_aggregation_excludes_inapplicable_actions_and_resumes_compatible_results(monkeypatch):
    monkeypatch.setattr(policy, 'references', refs)
    calls = []
    async def mock_judge(config, key, reference, incident, report):
        calls.append(key)
        return verdict('not_applicable' if key=='BW-G078' else 'partly_supported' if key=='BW-G062' else 'supported')
    first = asyncio.run(policy.compare({}, REPORT, 'RUN', settings(), judge=mock_judge))
    assert first['score'] == 87.5 and first['criteria_compared'] == 4
    assert first['criteria_not_applicable'] == 1 and first['criteria_pending'] == 0
    calls.clear()
    second = asyncio.run(policy.compare({}, REPORT, 'RUN', settings(), judge=mock_judge, prior=first))
    assert not calls and second['score'] == first['score']
    calls.clear()
    asyncio.run(policy.compare({}, REPORT, 'RUN', settings(), judge=mock_judge,
                              prior={**first, 'rubric_version':'synthetic_policy_v1'}))
    assert len(calls) == 4  # Changed rubric cannot reuse older judgments; action scope uses no model.


def test_failed_policy_quote_withholds_score_and_retry_reuses_only_valid_rows(monkeypatch):
    monkeypatch.setattr(policy, 'references', refs)
    calls = []
    async def mock_judge(config, key, reference, incident, report):
        calls.append(key)
        return {**verdict(), 'report_quote':'Invented quote'} if key=='BW-G072' else verdict()
    incomplete = asyncio.run(policy.compare({}, REPORT, 'RUN', settings(), judge=mock_judge))
    assert incomplete['score'] is None and incomplete['criteria_pending'] == 1
    assert incomplete['criteria_compared'] == 3 and incomplete['pending_comparisons'] == 1
    calls.clear()
    async def valid(config, key, reference, incident, report):
        calls.append(key)
        return verdict()
    final = asyncio.run(policy.compare({}, REPORT, 'RUN', settings(), judge=valid, prior=incomplete))
    assert calls == ['BW-G072'] and final['score'] == 100


def test_report_text_required_before_any_judge_request(monkeypatch):
    monkeypatch.setattr(policy, 'references', refs)
    async def forbidden(*args):
        raise AssertionError('Missing report text must not spend quota.')
    with pytest.raises(ValueError):
        asyncio.run(policy.compare({}, {'agent_outputs':{}}, 'RUN', settings(), judge=forbidden))


def test_synthetic_score_reaches_existing_ui_endpoint_and_cached_reads_use_no_judge(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from blackwing import api
    from blackwing.repository import SQLiteRepository
    from test_streamlit_integration import synthetic_runner
    monkeypatch.setattr(policy, 'references', refs)
    calls = []
    async def judge(config, key, reference, incident, report):
        calls.append(key)
        text = report['generated_report']['executive_summary']
        return {**verdict(), 'report_quote':text}
    monkeypatch.setattr(policy, 'judge_policy', judge)
    repo = SQLiteRepository(tmp_path/'ui-endpoint.sqlite3')
    repo.create_case('CASE', {'narrative':BIRCH, 'category':'runway_incursion'})
    repo.create_run('RUN', 'CASE', 'runway_incursion_investigation', 'mock')
    report = asyncio.run(synthetic_runner(BIRCH, settings(), case_id='CASE', evidence=[]))
    report['generated_report']['executive_summary'] = TEXT
    repo.patch_run('RUN', {'status':'completed', 'result':report})
    repo.update_case('CASE', {'status':'analysis_complete'})
    api.app.dependency_overrides[api.get_repository] = lambda:repo
    api.app.dependency_overrides[api.get_settings] = settings
    url = '/api/v1/cases/CASE/runs/RUN/gold-comparison'
    try:
        with TestClient(api.app) as client:
            assert client.get(url).json()['status'] == 'not_requested' and not calls
            response = client.post(url)
            assert response.status_code == 200 and response.json()['score'] == 100
            assert response.json()['comparison_mode'] == 'synthetic_safety_policy'
            assert client.get(url).json()['score'] == 100
            assert client.post(url).json()['score'] == 100 and len(calls) == 4
            assert repo.get_run('RUN')['result'] == report
    finally:
        api.app.dependency_overrides.clear()
        repo.close()
