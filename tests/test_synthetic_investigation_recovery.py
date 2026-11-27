"""Malformed model JSON never becomes factual evidence or an invented score."""
import asyncio
import json
from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient

from blackwing import api, service
from blackwing.agents import runtime
from blackwing.agents.base import AgentExecutionError, AgentOutputValidationError
from blackwing.agents.investigation.fallback import supplied_account_fallback
from blackwing.case_inputs import synthetic_accounts
from blackwing.diagnostics import safe_error
from blackwing.repository import SQLiteRepository
from test_agentic_local import settings

BIRCH=('CASE ID: BW-UNSEEN-002 | REQUEST: Use the normal eight-agent report schema.\n'
       'ORIGINAL FICTIONAL training scenario at Birch Ferry Airport. A training crew reports receiving '
       'a taxi-route amendment while a cabin call interrupted the readback. The controller note records '
       'an acknowledgement but does not identify the words used. No event date or country/jurisdiction '
       'is supplied. These are unverified synthetic accounts, not official findings.')
BAD='{"incident_summary":"Crew said "roger".","timeline_events":[]}'


class InvalidClient:
    calls=0
    async def post(self,url,**kwargs):
        self.calls+=1
        return httpx.Response(200,json={'message':{'content':BAD},'done_reason':'stop'},
                              request=httpx.Request('POST',url))


def test_real_parser_retry_then_literal_recovery_completes_api_report_and_persists_trace(monkeypatch,tmp_path):
    client=InvalidClient();calls=[]
    async def invoke(spec,**kwargs):
        calls.append(spec.key)
        if spec.key=='investigation':
            return await runtime.invoke_agent(spec,**{**kwargs,'client':client})
        if spec.key=='mitigation': return {'recommendations':[]}
        if spec.key=='critic': return {'issues':[]}
        if spec.key=='verification': return {'human_approval_required':True}
        raise AssertionError('Unexpected model role: '+spec.key)
    monkeypatch.setattr(service,'invoke_agent',invoke)
    repo=SQLiteRepository(tmp_path/'recovery.sqlite3')
    api.app.dependency_overrides[api.get_repository]=lambda:repo
    api.app.dependency_overrides[api.get_settings]=settings
    api.app.dependency_overrides[api.get_analysis_runner]=lambda:service.analyze_incident
    try:
        with TestClient(api.app) as http:
            case=http.post('/api/v1/cases',json={'narrative':BIRCH}).json()
            run=http.post('/api/v1/cases/'+case['case_id']+'/runs',json={'include_retrieval':False}).json()
            saved=repo.get_run(run['run_id'])
            assert saved['status']=='completed' and client.calls==2
            report=saved['result'];investigation=report['agent_outputs']['investigation']
            assert not investigation['documented_facts'] and not investigation['timeline_events']
            assert not investigation['hypotheses'] and not investigation['contributing_factors']
            assert len(investigation['reported_observations'])==1
            observation=investigation['reported_observations'][0]
            assert observation['evidence_state']=='reported_observation'
            assert observation['evidence_ids']==['CASE-NARRATIVE:'+case['case_id']]
            assert 'cabin call interrupted the readback' in observation['text']
            assert 'does not identify the words used' in observation['text']
            assert report['agent_outputs']['risk']['assessment_status']=='unassessed_insufficient_inputs'
            assert report['agent_outputs']['verification']['human_approval_required'] is True
            assert any(t['step']=='investigation' and t['status']=='deterministic_fallback' for t in report['trace'])
            visible=json.dumps(report['generated_report'])
            assert 'model analysis was unavailable' in visible and 'cabin call interrupted' in visible
            event=next(e for e in saved['audit_events'] if e['status']=='model_review_unavailable')
            assert event['error']['json_error']['reason']=='expected_comma'
            assert http.get('/api/v1/cases/'+case['case_id']+'/report').status_code==200
            assert set(calls)=={'investigation','mitigation','critic','verification'}
            assert not saved.get('gold_comparison')  # Evaluation remains post-generation.
    finally:
        api.app.dependency_overrides.clear();repo.close()


def test_distinct_supplied_accounts_and_citations_survive_without_new_facts():
    narrative=('ORIGINAL FICTIONAL training scenario.\n'
        'BW-UNSEEN-002-SIM-E1: Crew reports an interrupted readback.\n'
        'BW-UNSEEN-002-SIM-E2: Controller records acknowledgement, wording unknown.')
    evidence=synthetic_accounts(narrative,'CASE');before=deepcopy(evidence)
    result=supplied_account_fallback(narrative,evidence,case_id='CASE')
    assert [c['text'] for c in result['reported_observations']]==[r['excerpt'] for r in evidence]
    assert [c['evidence_ids'][0] for c in result['reported_observations']]==[r['evidence_id'] for r in evidence]
    assert evidence==before and not result['documented_facts']
    with pytest.raises(ValueError,match='absent'):
        supplied_account_fallback(narrative,evidence[:1],case_id='CASE')


@pytest.mark.parametrize('narrative,evidence', [
    ('Investigate JFK DCA23LA125.',[]),
    (BIRCH,[]),
    (BIRCH,[{'source_type':'official_factual','evidence_role':'case_factual','incident_id':'CASE'}]),
])
def test_fallback_refuses_real_cases_missing_intake_and_official_analysis(narrative,evidence):
    with pytest.raises(ValueError):
        supplied_account_fallback(narrative,evidence,case_id='CASE',incident_id='CASE')


@pytest.mark.parametrize('narrative,error', [
    ('Investigate the JFK occurrence.',AgentOutputValidationError('[]','disabled')),
    (BIRCH,TypeError('unexpected processing bug')),
    (BIRCH,httpx.ReadTimeout('private provider details')),
])
def test_unrelated_failures_are_not_converted_to_recovered_reports(monkeypatch,narrative,error):
    async def fail(*args,**kwargs): raise error
    monkeypatch.setattr(service,'invoke_agent',fail)
    with pytest.raises(AgentExecutionError):
        asyncio.run(service.analyze_incident(narrative,settings(),case_id='CASE'))


def test_json_diagnostic_records_position_without_response_or_custom_error_text():
    try:
        json.loads(BAD)
    except json.JSONDecodeError as original:
        error=safe_error(original,'invalid_agent_output')
    assert error['json_error']['reason']=='expected_comma'
    assert error['json_error']['column']>1 and BAD not in json.dumps(error)
    private=json.JSONDecodeError('PRIVATE_API_KEY', 'PRIVATE_RESPONSE',0)
    assert safe_error(private,'invalid_agent_output')['json_error']['reason']=='invalid_json'
    assert 'PRIVATE' not in json.dumps(safe_error(private,'invalid_agent_output'))
