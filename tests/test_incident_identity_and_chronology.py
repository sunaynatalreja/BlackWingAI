"""Incident identity, chronological context and categorical risk regressions."""
import asyncio
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from blackwing.incident_identity import resolve_incident_reference
from blackwing.agents.risk.agent import normalize_model_payload
from blackwing.agents.risk.evidence import ground_risk_output
from blackwing.agents.report.workflow import _fallback_section
from blackwing.rag.agent_router import _diverse_current
from test_streamlit_integration import backend, synthetic_runner


@pytest.mark.parametrize('narrative,explicit,expected', [
    ('Investigate DCA23LA133; reference DCA23LA133.', None, 'DCA23LA133'),
    ('Compare DCA23LA133 with DCA23LA125.', None, None),
    ('A fictional test referencing DCA23LA133.', None, None),
    ('Investigate DCA23LA133.', 'DCA23LA125', 'DCA23LA125'),
    ('A runway crossing at an unspecified airport.', None, None),
])
def test_explicit_reference_is_unique_and_not_guessed(narrative, explicit, expected):
    assert resolve_incident_reference(narrative, explicit) == expected


def test_intake_forwards_all_case_metadata_and_preserves_new_report_tabs(backend):
    from blackwing import copilot
    case=copilot.create_case(narrative="Investigate Honolulu runway sequence DCA23LA133.", location="Honolulu HNL",
        jurisdiction="US", incident_date="2023-01-23", source_incident_id="DCA23LA133")
    assert case["source_incident_id"]=="DCA23LA133" and case["incident_date"]=="2023-01-23"
    assert case["location"]=="Honolulu HNL" and case["jurisdiction"]=="US"
    ui=Path(__file__).resolve().parents[1]/"blackwing/ui.py"
    at=AppTest.from_file(str(ui),default_timeout=15).run()
    assert not at.exception
    assert at.text_area(key="incident_narrative_custom").label=="Incident narrative"
    assert at.file_uploader(key="source-upload").label=="Attach source report (optional)"
    assert at.button(key="analyze-incident").label=="Analyze incident"
    assert "home-new" not in [b.key for b in at.button]
    assert not backend.runs


def test_legacy_saved_case_resolves_reference_before_agent_retrieval(monkeypatch, backend):
    from blackwing import api
    from test_agentic_local import settings
    record = backend.create_case('legacy-case', {'narrative':'Investigate the Honolulu sequence DCA23LA133.',
        'source_incident_id':None, 'jurisdiction':'US', 'incident_date':'2023-01-23', 'category':'runway_incursion'})
    backend.create_run('routing-run', 'legacy-case', 'runway_incursion_investigation', 'mock')
    calls = []
    class Retriever:
        def search(self, query, documents, **kwargs):
            calls.append(kwargs)
            return {'results':[]}
    captured = {}
    async def runner(narrative, settings, **kwargs):
        captured.update(kwargs)
        return await synthetic_runner(narrative, settings, **kwargs)
    monkeypatch.setattr(api, 'retriever', Retriever())
    monkeypatch.setenv('BLACKWING_AGENT_RAG', '1')
    asyncio.run(api._execute_run('routing-run', record, [], settings(), backend, runner))
    assert backend.get_run('routing-run')['status'] == 'completed'
    assert captured['source_incident_id'] == 'DCA23LA133'
    assert all(c['case_id'] == 'DCA23LA133' for c in calls if c['scope'] in {'direct','historical'})
    assert any('DCA23LA133' in str(e.get('detail')) for e in backend.get_run('routing-run')['audit_events'])
    assert backend.get_case('legacy-case')['source_incident_id'] is None  # Saved intake is immutable here.


def test_wider_packet_retains_untimed_continuation_and_spaced_clock():
    def row(page, chunk, text):
        return {'source_id':'FACT', 'chunk_id':chunk, 'page_number':page, 'excerpt':text}
    rows = [row(17,'context','Airport equipment background'),
        row(9,'hold','At 1609: 57, controller instructed hold short. Pilot readback.'),
        row(10,'cross','At 1610:21, controller instructed cross runway 4R.'),
        row(10,'correction','Controller corrected the turn and instructed hold short of runway 8L.'),
        row(11,'after','At 1615:23, cleared to cross runway 8L.'),
        row(8,'before','At 1607:57, cleared to land.')]
    selected = _diverse_current(rows, 5)
    assert {'hold','cross','correction','after'} <= {r['chunk_id'] for r in selected}
    assert all(r in rows for r in selected)


@pytest.mark.parametrize('severity,likelihood,index', [(3,'A','A3'), (1,'C','C1')])
def test_grounded_inputs_produce_categorical_index(severity,likelihood,index):
    evidence = [{'chunk_id':'FACT:1', 'source_type':'official_factual', 'evidence_role':'case_factual'}]
    result = ground_risk_output({'hazards':[{'hazard_id':'H1','description':'Potential hazard',
        'evidence_ids':['FACT:1'], 'severity':severity, 'likelihood':likelihood,
        'severity_basis':'Verified severity basis supplied for test',
        'likelihood_basis':'Verified occurrence/exposure basis supplied for test'}]}, evidence)
    assert result['hazards'][0]['risk_index'] == index
    assert result['numeric_score'] is None


def test_prompt_alone_and_numeric_legacy_ratings_cannot_produce_index():
    evidence = [{'evidence_id':'CASE-NARRATIVE:case', 'source_type':'user_submitted', 'evidence_role':'reported'}]
    result = ground_risk_output({'hazards':[{'hazard_id':'H1','description':'Potential hazard',
        'evidence_ids':['CASE-NARRATIVE:case'], 'severity':3, 'likelihood':'C',
        'severity_basis':'Prompt assertion', 'likelihood_basis':'Prompt assertion'}]}, evidence)
    assert result['hazards'][0]['risk_index'] is None
    assert result['assessment_status'] == 'unassessed_insufficient_inputs'
    with pytest.raises(ValidationError):
        normalize_model_payload({'hazards':[{'hazard_id':'H1','description':'Legacy', 'severity':3, 'likelihood':3}]})


def test_risk_model_scalar_list_fields_are_repaired():
    out = normalize_model_payload({'hazards':[{'hazard_id':'H1','description':'Potential hazard',
        'evidence_ids':'PASSAGE-1','existing_controls':'None documented','control_gaps':None,
        'severity':None,'likelihood':None,'severity_basis':'','likelihood_basis':''}],
        'numeric_score':None})
    assert out['hazards'][0]['evidence_ids']==['PASSAGE-1']
    assert out['hazards'][0]['existing_controls']==['None documented']
    assert out['hazards'][0]['control_gaps']==[]


def test_report_assembly_preserves_timeline_time_and_categorical_risk():
    prior = {'investigation':{'timeline_events':[{'event':'Controller issued a clearance.',
        'time_reference':'1610:21', 'evidence_ids':['FACT:1'], 'evidence_state':'documented_fact'}]},
        'risk':{'numeric_score':None,'hazards':[{'risk_index':'C1','risk_level':'high',
            'scoring_status':'provisional_reference_only'}]}}
    overview = _fallback_section('overview', prior)
    assert overview['key_points'][0]['text'].startswith('1610:21:')
    assert overview['key_points'][0]['evidence_ids'] == ['FACT:1']
    summary = _fallback_section('risk_and_follow_up', prior)['summary']
    assert 'C1 (high)' in summary and '/25' not in summary and '5Ã—5' not in summary


def test_gold_judge_uses_shared_pacing_and_tokenizer(monkeypatch,tmp_path):
    import json
    import httpx
    from dataclasses import replace
    from blackwing import gold_comparison as gold
    from blackwing.agents import runtime
    from blackwing.rag import small_context
    from test_agentic_local import settings
    ca=tmp_path/'test-ca.crt'
    ca.write_text('fixture')
    monkeypatch.setenv('BLACKWING_LLM_CA_CERT',str(ca))
    captured={}
    async def post(client, **kwargs):
        captured.update(kwargs)
        return httpx.Response(200,request=httpx.Request('POST','https://mock.invalid'),json={
            'choices':[{'message':{'content':json.dumps({'status':'supported','gold_page':1,
                'gold_quote':'Controller issued a clearance.','rationale':'Matching supplied passage.'})}}]})
    original=httpx.AsyncClient
    monkeypatch.setattr(gold.httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(lambda req:pytest.fail('No network expected'))))
    monkeypatch.setattr(runtime,'_hosted_post',post)
    tokenizers=[]
    def count(messages,**kw):
        tokenizers.append(kw['tokenizer_model'])
        return 100
    monkeypatch.setattr(small_context,'count_input_tokens',count)
    cfg=replace(settings(),llm_provider='openai_compatible',api_min_interval_seconds=1.7)
    result=asyncio.run(gold.judge_claim(cfg,{'text':'Controller issued a clearance.'},
        {1:'Controller issued a clearance.'}))
    assert result['status']=='supported'
    assert captured['min_interval_seconds']==1.7 and tokenizers==[cfg.tokenizer_model]
    assert captured['payload']['max_tokens']<=1024
