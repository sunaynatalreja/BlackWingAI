from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from streamlit.testing.v1 import AppTest


def test_aes_gold_pdf_dependency_works_in_application_environment():
    writer=PdfWriter()
    writer.add_blank_page(width=200,height=200)
    writer.encrypt(user_password='', algorithm='AES-256')
    data=BytesIO();writer.write(data);data.seek(0)
    assert len(PdfReader(data).pages)==1


@pytest.mark.parametrize('reference,expected',[(True,'Seen'),(False,'Unseen'),(None,'Unseen')])
def test_report_case_match_and_categorical_risk_are_independent(reference,expected):
    report={'agent_outputs':{'similar_case':{'case_status':'unseen_case_with_similar_historical_cases'},
            'risk':{'hazards':[{'hazard_id':'H1','description':'Grounded hazard','severity':3,'likelihood':'C',
                'risk_index':'C3','risk_level':'medium','scoring_status':'provisional_reference_only'}]}}}
    script=f"from blackwing.ui import _render_report\n_render_report({{'sections':[]}}, {report!r}, 'case', 'run', reference_case={reference!r})"
    at=AppTest.from_string(script).run()
    assert not at.exception
    metrics={m.label:m.value for m in at.metric}
    assert metrics['Case match']==expected and metrics['Aviation risk']=='C3'
    df=at.dataframe[0].value
    assert df.iloc[0]['Severity']=='3 — major' and df.iloc[0]['Likelihood']=='C — remote'


def test_unassessed_report_does_not_fabricate_risk_rating():
    report={'agent_outputs':{'risk':{'hazards':[{'hazard_id':'H1','description':'Hypothesis',
        'scoring_status':'unassessed_insufficient_inputs','risk_index':None,'severity':None,'likelihood':None}]}}}
    at=AppTest.from_string(f"from blackwing.ui import _render_report\n_render_report({{'sections':[]}}, {report!r}, 'case','run',reference_case=True)").run()
    assert not at.exception
    assert {m.label:m.value for m in at.metric}['Aviation risk']=='Unassessed'
    assert at.dataframe[0].value.iloc[0]['Risk index']=='Unassessed'


def test_exact_incident_metadata_marks_seen_but_historical_analogy_does_not():
    from blackwing.agents.similar_case.agent import build_output
    current = {'incident_id':'DCA17IA148','source_type':'official_factual',
               'evidence_role':'case_factual','source_id':'NTSB-DCA17IA148',
               'document_type':'incident_report','excerpt':'Current factual record.'}
    exact = build_output('SFO runway overflight', [current], current_incident_id='DCA17IA148')
    assert exact['case_status'] == 'existing_case_match'
    analogy = dict(current, incident_id='DCA16IA001', evidence_role='historical_analogy')
    unseen = build_output('SFO runway overflight', [analogy], current_incident_id='DCA17IA148')
    assert unseen['case_status'] != 'existing_case_match'


def test_pdf_dependency_failure_is_sanitized_and_durable(monkeypatch,tmp_path):
    from fastapi.testclient import TestClient
    from pypdf.errors import DependencyError
    from blackwing import api,gold_comparison as gold
    from blackwing.repository import SQLiteRepository
    from test_agentic_local import settings
    from test_streamlit_integration import synthetic_runner
    repo=SQLiteRepository(tmp_path/'backend.sqlite3')
    api.app.dependency_overrides[api.get_repository]=lambda:repo
    api.app.dependency_overrides[api.get_settings]=settings
    api.app.dependency_overrides[api.get_analysis_runner]=lambda:synthetic_runner
    monkeypatch.setattr(gold,'gold_path',lambda item:tmp_path/'fixture.pdf')
    async def fail(*args):raise DependencyError('simulated private path and provider credential')
    monkeypatch.setattr(gold,'compare_report',fail)
    try:
        with TestClient(api.app) as client:
            case=client.post('/api/v1/cases',json={'narrative':'Investigate JFK runway conflict DCA23LA125.','source_incident_id':'DCA23LA125'}).json()
            run=client.post(f"/api/v1/cases/{case['case_id']}/runs",json={'include_retrieval':False}).json()
            url=f"/api/v1/cases/{case['case_id']}/runs/{run['run_id']}/gold-comparison"
            reply=client.post(url)
            assert reply.status_code==503
            assert reply.json()['detail']['code']=='gold_pdf_crypto_missing'
            assert 'private path' not in reply.text and 'credential' not in reply.text
            assert client.get(url).json()['last_error']['code']=='gold_pdf_crypto_missing'
            assert client.get('/health').json()['status']=='ok'
            assert client.get(f"/api/v1/cases/{case['case_id']}/report").status_code==200
    finally:
        api.app.dependency_overrides.clear();repo.close()


def test_private_token_usage_does_not_break_report_or_public_trace(tmp_path):
    from fastapi.testclient import TestClient
    from blackwing import api
    from blackwing.repository import SQLiteRepository
    from test_agentic_local import settings
    from test_streamlit_integration import synthetic_runner
    repo=SQLiteRepository(tmp_path/'token-usage.sqlite3')
    api.app.dependency_overrides[api.get_repository]=lambda:repo
    api.app.dependency_overrides[api.get_settings]=settings
    api.app.dependency_overrides[api.get_analysis_runner]=lambda:synthetic_runner
    try:
        with TestClient(api.app) as client:
            case=client.post('/api/v1/cases',json={'narrative':'Synthetic runway conflict report schema test.'}).json()
            run=client.post(f"/api/v1/cases/{case['case_id']}/runs",json={'include_retrieval':False}).json()
            saved=repo.get_run(run['run_id'])
            result=saved['result']
            result['trace']=[{**event,'prompt_tokens':42,'completion_tokens':7} for event in result['trace']]
            repo.patch_run(run['run_id'],{'result':result})
            report=client.get(f"/api/v1/cases/{case['case_id']}/report")
            trace=client.get(f"/api/v1/runs/{run['run_id']}/trace")
            assert report.status_code==200
            assert trace.status_code==200
            assert all('prompt_tokens' not in event and 'completion_tokens' not in event
                       for event in report.json()['report']['trace'])
            assert all('prompt_tokens' not in event and 'completion_tokens' not in event
                       for event in trace.json()['events'])
    finally:
        api.app.dependency_overrides.clear();repo.close()


def test_ui_shows_actionable_gold_error_without_losing_case_match(monkeypatch):
    import httpx
    from blackwing import copilot
    monkeypatch.setattr(copilot,'get_report_evaluation',lambda *args:{'legacy_fallback':True})
    monkeypatch.setattr(copilot,'get_gold_comparison',lambda *args:{'status':'not_requested','gold_id':'BW-HO-001','gold_case':'JFK'})
    def fail(*args):
        response=httpx.Response(503,request=httpx.Request('POST','http://test/'),json={'detail':{'code':'gold_pdf_crypto_missing','message':'Install project requirements and restart FastAPI.'}})
        response.raise_for_status()
    monkeypatch.setattr(copilot,'compare_with_gold',fail)
    at=AppTest.from_string("from blackwing.copilot_ui import show_gold_comparison\nshow_gold_comparison({'incident':{'case_id':'case'},'report_run_id':'run'})").run()
    at.button[0].click().run()
    assert not at.exception and any('Install project requirements' in e.value for e in at.error)
