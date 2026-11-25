import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from blackwing import gold_comparison as gold


def test_nine_case_catalog_and_no_cross_case_or_synthetic_matching():
    assert len(gold.catalog()) == 9
    assert gold.match_gold({'source_incident_id':'DCA23LA125'})['gold_id'] == 'BW-HO-001'
    assert gold.match_gold({'source_incident_id':'DCA23FA149'})['gold_id'] == 'BW-HO-004'
    assert gold.match_gold({'location':'Delhi IGI','incident_date':'2013-10-02'})['gold_id'] == 'BW-HO-003'
    assert gold.match_gold({'source_incident_id':'DCA23LA125','narrative':'Fictional training scenario'}) is None
    assert gold.match_gold({'source_incident_id':'DCA23LA125','incident_date':'2023-02-04'}) is None
    assert gold.match_gold({'source_incident_id':'DCA23LA125','jurisdiction':'IN'}) is None
    assert gold.match_gold({'source_incident_id':'VT-BXL','incident_date':'2023-01-13'}) is None
    assert gold.match_gold({'location':'JFK'}) is None


def test_claim_scope_deduplicates_and_preserves_abstention_and_risk():
    report={'agent_outputs':{'investigation':{
        'documented_facts':[{'text':'A runway alert occurred.','evidence_state':'documented_fact'},
                            {'text':'Cause is unknown','evidence_state':'unknown'}],
        'hypotheses':[{'text':'Workload may have contributed','evidence_state':'hypothesis'}]},
        'risk':{'hazards':[{'risk_index':'C3'}]}},
        'generated_report':{'sections':[{'key_points':[{'text':'A runway alert occurred.','classification':'documented_fact'}]}]}}
    before=copy.deepcopy(report)
    assert len(gold.claims_for_comparison(report)) == 1
    assert report == before


def test_quotes_must_exist_on_claimed_supplied_page():
    pages={2:'The runway occupied alert was annunciated at 2044:42.'}
    assert gold.validate_judgment({'status':'supported','gold_page':2,'gold_quote':'runway occupied alert','rationale':'Same alert'},pages)
    with pytest.raises(ValueError):
        gold.validate_judgment({'status':'supported','gold_page':1,'gold_quote':'runway occupied alert','rationale':'Bad page'},pages)
    with pytest.raises(ValueError):
        gold.validate_judgment({'status':'contradicted','gold_page':2,'gold_quote':'invented quotation here','rationale':'Bad quote'},pages)


@pytest.fixture
def reference(monkeypatch,tmp_path):
    path=tmp_path/'gold.pdf'
    path.write_bytes(b'read-only evaluation fixture')
    item={'gold_id':'BW-HO-001','case_label':'JFK','filename':'gold.pdf','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(gold,'match_gold',lambda incident:item)
    monkeypatch.setattr(gold,'gold_path',lambda item:path)
    import pypdf
    monkeypatch.setattr(pypdf,'PdfReader',lambda path:SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda:'An aircraft crossed runway 4L. A runway occupied alert occurred.')]))
    return path


def report_with_facts(*claims):
    return {'agent_outputs':{'investigation':{'documented_facts':[{'text':c,'evidence_state':'documented_fact'} for c in claims]}}}


def test_score_is_separate_quote_validated_and_source_is_unchanged(reference):
    before=reference.read_bytes()
    report=report_with_facts('An aircraft crossed runway 4L.','A runway occupied alert occurred.')
    settings=SimpleNamespace(api_model='mock')
    async def judge(settings,claim,pages):
        return {'status':'supported' if claim.startswith('An aircraft') else 'partly_supported',
            'gold_page':1,'gold_quote':'aircraft crossed runway 4L' if claim.startswith('An aircraft') else 'runway occupied alert occurred',
            'rationale':'Compare the stated occurrence.'}
    result=asyncio.run(gold.compare_report({},report,'run-1',settings,judge=judge))
    assert result['score']==75
    assert result['label']=='Gold comparison score'
    assert result['human_review_required'] is True
    assert result['run_id']=='run-1' and len(result['improvements'])==1
    assert reference.read_bytes()==before


def test_unverified_reference_support_is_pending_not_a_zero_score(reference):
    async def judge(settings,claim,pages):
        return {'status':'not_verified','gold_page':None,'gold_quote':'',
                'rationale':'The selected reference excerpt does not establish this action.'}
    result=asyncio.run(gold.compare_report({},report_with_facts('The aircraft crossed the runway.'),
        'run',SimpleNamespace(api_model='mock'),judge=judge))
    assert result['score'] is None and result['status']=='evaluation_incomplete'
    assert result['pending_comparisons']==1 and result['comparisons'][0]['points'] is None


def test_empty_claims_and_invalid_model_quote_do_not_get_numeric_scores(reference):
    settings=SimpleNamespace(api_model='mock')
    async def invalid(settings,claim,pages):
        return {'status':'supported','gold_page':1,'gold_quote':'Fabricated proof of agreement','rationale':'Invalid'}
    empty=asyncio.run(gold.compare_report({},report_with_facts(),'run',settings,judge=invalid))
    assert empty['score'] is None and empty['status']=='not_assessable'
    invalid_result=asyncio.run(gold.compare_report({},report_with_facts('Runway crossing'),'run',settings,judge=invalid))
    assert invalid_result['score'] is None and invalid_result['status']=='evaluation_incomplete'


def test_post_generation_api_cache_bound_to_run_and_report(monkeypatch,tmp_path):
    from fastapi.testclient import TestClient
    from blackwing import api
    from blackwing.repository import SQLiteRepository
    from test_agentic_local import settings
    from test_streamlit_integration import synthetic_runner
    repo=SQLiteRepository(tmp_path/'cases.sqlite3')
    api.app.dependency_overrides[api.get_repository]=lambda:repo
    api.app.dependency_overrides[api.get_settings]=settings
    api.app.dependency_overrides[api.get_analysis_runner]=lambda:synthetic_runner
    monkeypatch.setattr(gold,'gold_path',lambda item:tmp_path/'fixture.pdf')
    calls=[]
    async def compare(case,report,run_id,settings):
        calls.append(run_id)
        item=gold.match_gold(case)
        return {'status':'proposed','score':50,'label':'Gold comparison score','report_hash':gold.report_hash(report),
            'gold_sha256':item['sha256'],'rubric_version':gold.VERSION,'run_id':run_id,'human_review_required':True}
    monkeypatch.setattr(gold,'compare_report',compare)
    try:
        with TestClient(api.app) as client:
            case=client.post('/api/v1/cases',json={'source_incident_id':'DCA23LA125','narrative':'Investigate the reported JFK runway conflict.'}).json()
            run=client.post(f"/api/v1/cases/{case['case_id']}/runs",json={'include_retrieval':False}).json()
            url=f"/api/v1/cases/{case['case_id']}/runs/{run['run_id']}/gold-comparison"
            assert client.get(url).json()['status']=='not_requested'
            before=copy.deepcopy(repo.get_run(run['run_id'])['result'])
            assert client.post(url).json()['score']==50
            assert client.post(url).json()['score']==50
            assert len(calls)==1 and repo.get_run(run['run_id'])['result']==before
            other=client.post('/api/v1/cases',json={'narrative':'Another incident with no verified case match.'}).json()
            assert client.get(url.replace(case['case_id'],other['case_id'])).status_code==404
        # Persisted score uses the existing repository, not a session database.
        repo.close()
        reopened=SQLiteRepository(tmp_path/'cases.sqlite3')
        assert reopened.get_run(run['run_id'])['gold_comparison']['score']==50
        reopened.close()
    finally:
        api.app.dependency_overrides.clear()


def test_report_panel_shows_only_exact_label_and_no_automatic_model_call(monkeypatch):
    from blackwing import copilot
    monkeypatch.setattr(copilot,'get_report_evaluation',lambda *args:{'legacy_fallback':True})
    monkeypatch.setattr(copilot,'get_gold_comparison',lambda *args:{'status':'proposed','score':75,
        'gold_case':'JFK','gold_id':'BW-HO-001','notice':'Human review required.','comparisons':[], 'factual_claims':2})
    def forbidden(*args): raise AssertionError('No model call on display')
    monkeypatch.setattr(copilot,'compare_with_gold',forbidden)
    script="from blackwing.copilot_ui import show_gold_comparison\nshow_gold_comparison({'incident':{'case_id':'case'},'report_run_id':'run'})"
    at=AppTest.from_string(script).run()
    assert not at.exception
    assert at.metric[0].label=='Gold comparison score' and at.metric[0].value=='75/100'


def test_unmatched_panel_has_no_compare_button(monkeypatch):
    from blackwing import copilot
    monkeypatch.setattr(copilot,'get_report_evaluation',lambda *args:{'legacy_fallback':True})
    monkeypatch.setattr(copilot,'get_gold_comparison',lambda *args:{'status':'not_applicable','score':None,'reason':'Fictional case has no matching gold report.'})
    at=AppTest.from_string("from blackwing.copilot_ui import show_gold_comparison\nshow_gold_comparison({'incident':{'case_id':'case'},'report_run_id':'run'})").run()
    assert not at.exception and not at.button
    assert at.metric[0].value=='Not applicable'
