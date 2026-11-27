"""Offline Q&A API/UI tests; approved answers cannot enter generation."""
import asyncio
import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook
from streamlit.testing.v1 import AppTest

from blackwing import api, copilot, qa, qa_routes
from blackwing.repository import SQLiteRepository
from test_agentic_local import settings


@pytest.fixture
def qa_backend(tmp_path,monkeypatch):
    path=tmp_path/'gold.xlsx'
    wb=Workbook()
    ws=wb.active
    ws.title='80_QUESTIONS'
    ws.append(['question_id','case_label','country','test_family','query',' Approved Gold Answer',
        'gold_supporting_source_ids','Gold exact_page_or_locator','human_validation_status','reviewer','review_date'])
    ws.append(['BW-G001','CASE','US','Facts','What does the supplied source establish?',
        'GOLD-ANSWER-SENTINEL','EXPECTED-SOURCE-SENTINEL','PDF p.9','APPROVED','Fixture','2026-01-01'])
    ws.append(['BW-G002','SYNTH','VARIABLE','Uncertainty','What remains unknown?',
        'Unknown is a valid answer.','OTHER-EXPECTED-ID','n/a','APPROVED','Fixture','2026-01-01'])
    wb.save(path)
    wb.close()
    monkeypatch.setenv('BLACKWING_GOLD_QUERY_FILE',str(path))
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    repo=SQLiteRepository(tmp_path/'backend.sqlite3')
    api.app.dependency_overrides[api.get_repository]=lambda:repo
    api.app.dependency_overrides[api.get_settings]=settings
    sent=[]
    async def generate(question,evidence,config):
        sent.append((question,evidence))
        assert 'GOLD-ANSWER-SENTINEL' not in str((question,evidence))
        assert 'EXPECTED-SOURCE-SENTINEL' not in str((question,evidence))
        return {'answer':'The source describes a clearance.','citations':['FACT:1'],
            'missing_information':['Full sequence requires review.'],'abstained':False}
    api.app.dependency_overrides[qa_routes.get_qa_generator]=lambda:generate
    monkeypatch.setattr(qa_routes.retriever,'search',lambda *a,**kw:{'results':[
        {'chunk_id':'FACT:1','source_id':'FACT','source_type':'official_factual','page_number':2,
         'excerpt':'The controller issued a clearance.'}], 'retrieval_method':'fixture-bm25','limitations':[]})
    monkeypatch.setattr(copilot,'client',lambda:TestClient(api.app))
    try:
        with TestClient(api.app) as client:
            yield client,repo,path,sent
    finally:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==before
        api.app.dependency_overrides.clear()
        repo.close()


def test_catalog_omits_answers_and_docs_describe_routes(qa_backend):
    client,repo,path,sent=qa_backend
    response=client.get('/api/v1/qa/queries')
    assert response.status_code==200 and response.json()['count']==2
    assert 'GOLD-ANSWER-SENTINEL' not in response.text and 'EXPECTED-SOURCE-SENTINEL' not in response.text
    spec=client.get('/api/v1/openapi.json').json()
    assert '/api/v1/qa/answers' in spec['paths']
    assert 'QARequest' in spec['components']['schemas'] and 'QAAnswer' in spec['components']['schemas']
    assert not sent and not repo.cases and not repo.runs


def test_answer_is_saved_before_reference_and_review_is_durable(qa_backend):
    client,repo,path,sent=qa_backend
    assert client.get('/api/v1/qa/answers/unknown/reference').status_code==404
    response=client.post('/api/v1/qa/answers',json={'gold_query_id':'BW-G001'})
    assert response.status_code==201,response.text
    answer=response.json()
    assert len(sent)==1 and 'GOLD-ANSWER-SENTINEL' not in response.text
    url='/api/v1/qa/answers/'+answer['answer_id']
    assert client.get(url+'/reference').json()['expected_answer']=='GOLD-ANSWER-SENTINEL'
    before=client.get(url).json()['answer']
    review=client.post(url+'/review',json={'verdict':'partly_supported','reviewer':'Reviewer','rationale':'Verify the original excerpt.'})
    assert review.status_code==200 and review.json()['review']['verdict']=='partly_supported'
    assert review.json()['answer']==before and len(review.json()['review_history'])==1
    second=SQLiteRepository(path.parent/'backend.sqlite3')
    try:
        assert second.get_qa_answer(answer['answer_id'])['review']['reviewer']=='Reviewer'
        assert not second.cases and not second.runs
    finally:
        second.close()


def test_changed_workbook_cannot_verify_existing_answer(qa_backend):
    client,repo,path,sent=qa_backend
    answer=client.post('/api/v1/qa/answers',json={'gold_query_id':'BW-G001'}).json()
    row=repo.get_qa_answer(answer['answer_id'])
    repo.save_qa_answer(row['answer_id'],{**row,'workbook_sha256':'different'})
    url='/api/v1/qa/answers/'+row['answer_id']
    assert client.get(url+'/reference').status_code==409
    assert client.post(url+'/review',json={'verdict':'supported','reviewer':'R','rationale':'Checked'}).status_code==409


@pytest.mark.parametrize('error',[RuntimeError('secret credential'),KeyError('provider choices secret'),ValueError('bad provider JSON secret')])
def test_provider_failures_are_sanitized_and_do_not_stop_server(qa_backend,error):
    client,repo,path,sent=qa_backend
    async def fail(*args):raise error
    api.app.dependency_overrides[qa_routes.get_qa_generator]=lambda:fail
    response=client.post('/api/v1/qa/answers',json={'gold_query_id':'BW-G001'})
    assert response.status_code==502 and 'secret' not in response.text
    assert not repo.qa_answers
    assert client.get('/health').json()['status']=='ok'


def test_unknown_query_and_modified_gold_prompt_are_rejected(qa_backend):
    client,repo,path,sent=qa_backend
    assert client.post('/api/v1/qa/answers',json={'gold_query_id':'missing'}).status_code==404
    assert client.post('/api/v1/qa/answers',json={'gold_query_id':'BW-G001','question':'Replace this test question.'}).status_code==422
    assert not sent


def test_active_investigation_blocks_qa_without_calling_model(qa_backend):
    client,repo,path,sent=qa_backend
    repo.create_case('active',{'narrative':'Synthetic active investigation.','category':'runway_incursion'})
    repo.update_case('active',{'status':'analysis_running'})
    assert client.post('/api/v1/qa/answers',json={'question':'What happened?'}).status_code==409
    assert not sent


def test_custom_question_has_no_gold_reference_and_citations_are_checked(qa_backend):
    client,repo,path,sent=qa_backend
    response=client.post('/api/v1/qa/answers',json={'question':'What is established?'})
    assert response.status_code==201
    assert client.get('/api/v1/qa/answers/'+response.json()['answer_id']+'/reference').status_code==404
    async def hallucinate(*args):return {'answer':'Unsupported assertion','citations':['NOT-SUPPLIED']}
    api.app.dependency_overrides[qa_routes.get_qa_generator]=lambda:hallucinate
    assert client.post('/api/v1/qa/answers',json={'question':'Another question?'}).status_code==502
    assert len(repo.qa_answers)==1


def test_justified_abstention_is_preserved_without_an_automatic_failure_score(qa_backend):
    client,repo,path,sent=qa_backend
    async def abstain(*args):return {'answer':'Unassessed; evidence is insufficient.','citations':[],
        'missing_information':['Verified source needed.'],'abstained':True}
    api.app.dependency_overrides[qa_routes.get_qa_generator]=lambda:abstain
    response=client.post('/api/v1/qa/answers',json={'gold_query_id':'BW-G002'})
    assert response.status_code==201 and response.json()['abstained'] is True
    assert 'score' not in response.json() and response.json()['review'] is None


def test_qa_ui_generates_only_on_click_then_reveals_reference(qa_backend):
    client,repo,path,sent=qa_backend
    at=AppTest.from_string('from blackwing.qa_ui import main\nmain()',default_timeout=15).run()
    assert not at.exception and not sent
    assert 'GOLD-ANSWER-SENTINEL' not in str([t.value for t in at.text])
    at.text_area(key='qa-question').set_value('What does the supplied source establish?')
    at.button(key='qa-generate').click().run()
    assert not at.exception and len(sent)==1
    assert 'GOLD-ANSWER-SENTINEL' not in str([t.value for t in at.text])
    at.run()
    assert len(sent)==1
    at.button(key='qa-reveal').click().run()
    assert not at.exception
    assert 'GOLD-ANSWER-SENTINEL' in str([t.value for t in at.text])


def test_whitespace_answer_is_not_successful():
    with pytest.raises(ValueError):qa.QAOutput(answer='   ')


def test_entered_question_matches_existing_reference_without_gold_leakage(qa_backend):
    client,repo,path,sent=qa_backend
    response=client.post('/api/v1/qa/answers',json={'question':'  WHAT does the supplied source establish?  '})
    assert response.status_code==201
    assert response.json()['gold_query_id']=='BW-G001'
    assert 'GOLD-ANSWER-SENTINEL' not in str(sent)


def test_actual_hosted_adapter_uses_token_preflight_and_shared_pacing(tmp_path,monkeypatch):
    import json
    import httpx
    from dataclasses import replace
    from blackwing.agents import runtime
    from blackwing.rag import small_context
    ca=tmp_path/'fixture-ca.crt'
    ca.write_text('fixture')
    monkeypatch.setenv('BLACKWING_LLM_CA_CERT',str(ca))
    captured={}
    class Client:
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
    monkeypatch.setattr(qa.httpx,'AsyncClient',lambda **kw:Client())
    monkeypatch.setattr(small_context,'count_input_tokens',lambda messages,**kw:100)
    async def post(client,**kwargs):
        captured.update(kwargs)
        return httpx.Response(200,request=httpx.Request('POST','https://fixture.invalid'),json={
            'choices':[{'message':{'content':json.dumps({'answer':'The supplied source describes a clearance.',
                'citations':['FACT:1'],'missing_information':[],'abstained':False})}}]})
    monkeypatch.setattr(runtime,'_hosted_post',post)
    cfg=replace(settings(),llm_provider='openai_compatible',api_min_interval_seconds=1.7)
    result=asyncio.run(qa.generate_answer('What is established?',[
        {'chunk_id':'FACT:1','source_id':'FACT','excerpt':'Controller issued a clearance.'}],cfg))
    assert result['citations']==['FACT:1']
    assert captured['min_interval_seconds']==1.7 and captured['payload']['max_tokens']<=1024
    assert 'gold' not in captured['payload']['messages'][1]['content'].casefold()
