import asyncio
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from blackwing.evaluation import report_judge as judge
from blackwing.evaluation.materials import digest
from blackwing import evaluation_routes
from blackwing.repository import SQLiteRepository
from test_agentic_local import settings

FACT='The crew rejected takeoff before the intersection.'
REPORT={'generated_report':{'executive_summary':FACT,'sections':[]},
        'agent_outputs':{'investigation':{'documented_facts':[{'text':'Hidden agent text is not report text'}]}}}
SNAPSHOT={'complete':True,'evidence':[{'chunk_id':'INPUT:1','source_type':'official_factual',
    'evidence_role':'case_factual','excerpt':FACT}]}


@pytest.fixture
def materials(tmp_path):
    p=tmp_path/'reference.pdf';p.write_bytes(b'Immutable reference bytes')
    return {'reference_id':'REF','reference_path':str(p),'reference_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'anchors_sha256':'anchors','questions':[{'question_id':'Q1','question':'What did the crew do?',
            'expected_answer':FACT,'answerability':'judge_from_input'}],'pages':{'1':FACT}}


def verdict(status='correct'):
    return {'status':status,'report_field':'executive_summary','report_quote':FACT,
            'reference_page':'1','reference_quote':FACT,'input_answerability':'supported',
            'input_id':'E0','input_quote':FACT,'rationale':'The saved report and reference agree.'}


def test_only_visible_saved_report_text_is_judged(materials):
    packet=judge.prepare_packet(REPORT,materials['questions'][0],materials,SNAPSHOT)
    assert 'Hidden agent text' not in json.dumps(packet)
    assert packet['report_fields']['executive_summary']==FACT
    assert judge.validate_judgment(verdict(),packet)['points']==1
    for field in ['report_quote','reference_quote','input_quote']:
        bad=verdict();bad[field]='Fabricated source quotation'
        with pytest.raises(ValueError):judge.validate_judgment(bad,packet)


def test_null_unavailable_quotes_are_accepted_without_weakening_evidence_checks(materials):
    packet=judge.prepare_packet(REPORT,materials['questions'][0],materials,SNAPSHOT)
    value={**verdict(),'status':'requires_review','report_field':None,'report_quote':None,
           'reference_page':None,'reference_quote':None,'input_answerability':'unknown',
           'input_id':None,'input_quote':None}
    result=judge.validate_judgment(value,packet)
    assert result['report_quote']==result['reference_quote']==result['input_quote']==''
    assert result['points'] is None
    with pytest.raises(ValueError):
        judge.validate_judgment({**value,'status':'correct'},packet)


def test_validation_reports_safe_specific_categories(materials):
    packet=judge.prepare_packet(REPORT,materials['questions'][0],materials,SNAPSHOT)
    for change, code in [({'status':'invented'}, 'question_schema_status_invalid_choice'),
                         ({'report_quote':'fabricated quotation'}, 'invalid_report_quote_or_field'),
                         ({'reference_quote':'fabricated quotation'}, 'invalid_reference_quote_or_page'),
                         ({'input_quote':'fabricated quotation'}, 'invalid_supported_input_quote_or_role')]:
        with pytest.raises(judge.EvaluationError) as exc:
            judge.validate_judgment({**verdict(), **change},packet)
        assert exc.value.code == code
        assert 'fabricated' not in str(exc.value)


def test_unrelated_and_abstaining_quotations_withhold_points(materials):
    question={**materials['questions'][0], 'question':'What phases were the aircraft in?',
              'expected_answer':'GoAir was taking off and the IAF aircraft was taxiing in.'}
    unrelated='Historical incident date not verified from supplied metadata.'
    input_text='The GoAir and IAF aircraft were involved in a conflict at Delhi.'
    packet=judge.prepare_packet({'generated_report':{'executive_summary':unrelated}},question,
                                {**materials,'pages':{'1':'GoAir was taking off and the IAF aircraft was taxiing in.'}},
                                {'evidence':[{'excerpt':input_text,'evidence_role':'case_factual'}]})
    v={**verdict(),'report_quote':unrelated,'reference_quote':packet['reference_excerpts']['1'],
       'input_quote':input_text}
    result=judge.validate_judgment(v,packet)
    assert result['status']=='requires_review' and result['points'] is None
    assert result['input_grounding']=='passage_relevance_requires_review'
    assert 'input quotation' in result['review_reason']
    abstention='Exact sequence of events and clearance details are not established.'
    packet['report_fields']['executive_summary']=abstention
    result=judge.validate_judgment({**v,'status':'contradicted','report_quote':abstention},packet)
    assert result['status']=='requires_review' and result['points'] is None


def test_airport_and_date_are_not_discarded_as_generic_context(materials):
    text='A conflict was reported at Delhi IGI Airport on 2 October 2013.'
    question={**materials['questions'][0], 'expected_answer':'Delhi IGI Airport, 2 October 2013'}
    report={'generated_report':{'executive_summary':text}}
    snapshot={'evidence':[{'excerpt':text, 'evidence_role':'case_factual'}]}
    packet=judge.prepare_packet(report,question,{**materials,'pages':{'1':text}},snapshot)
    result=judge.validate_judgment({**verdict(),'report_quote':text,'reference_quote':text,'input_quote':text},packet)
    assert result['status']=='correct' and result['points']==1


def test_long_verbatim_reference_quote_within_limit_is_accepted(materials):
    text='The crew rejected takeoff before the intersection. ' * 5
    packet=judge.prepare_packet(REPORT,materials['questions'][0],{**materials,'pages':{'1':text}},SNAPSHOT)
    result=judge.validate_judgment({**verdict(),'reference_quote':text},packet)
    assert result['points']==1


def test_reference_page_is_corrected_only_for_unique_verbatim_match(materials):
    packet=judge.prepare_packet(REPORT,materials['questions'][0],materials,SNAPSHOT)
    packet['reference_excerpts']={'1':'Unrelated reference text goes here.', '2':FACT}
    result=judge.validate_judgment(verdict(),packet)
    assert result['reference_page']=='2' and result['reference_quote']==FACT
    packet['reference_excerpts']['3']=FACT
    with pytest.raises(judge.EvaluationError):
        judge.validate_judgment(verdict(),packet)


def test_report_field_is_corrected_only_for_unique_verbatim_match(materials):
    packet=judge.prepare_packet(REPORT,materials['questions'][0],materials,SNAPSHOT)
    packet['report_fields']={'executive_summary':'Different saved report passage.', 'sections.0.summary':FACT}
    result=judge.validate_judgment(verdict(),packet)
    assert result['report_field']=='sections.0.summary' and result['report_quote']==FACT
    packet['report_fields']['sections.1.summary']=FACT
    with pytest.raises(judge.EvaluationError):
        judge.validate_judgment(verdict(),packet)


def test_adapter_retains_original_status_for_review_explanation(monkeypatch, materials):
    question={**materials['questions'][0], 'expected_answer':'The crew rejected takeoff.'}
    packet=judge.prepare_packet(REPORT,question,materials,SNAPSHOT)
    packet['report_fields']['executive_summary']='The details are not established.'
    async def evaluate(settings, system, payload, validator):
        return validator({**verdict(), 'report_quote':'The details are not established.'})
    monkeypatch.setattr(judge,'validated_evaluation',evaluate)
    returned=asyncio.run(judge.judge_question(settings(),packet))
    result=judge.validate_judgment(returned,packet)
    assert result['status']=='requires_review' and 'uncertainty' in result['review_reason']


def test_justified_abstention_and_later_only_facts_do_not_earn_or_lose_coverage(materials):
    report={'generated_report':{'executive_summary':'The minimum separation is unknown.'}}
    q={**materials['questions'][0],'answerability':'not_answerable_from_input'}
    packet=judge.prepare_packet(report,q,materials,SNAPSHOT)
    value={**verdict(),'status':'justified_abstention','report_quote':'The minimum separation is unknown.',
           'input_answerability':'not_supported','input_id':None,'input_quote':''}
    result=judge.validate_judgment(value,packet)
    assert result['points'] is None and result['status']=='justified_abstention'
    missing={**value,'status':'missing','report_field':None,'report_quote':''}
    assert judge.validate_judgment(missing,packet)['status']=='not_applicable'
    packet=judge.prepare_packet(REPORT,q,materials,SNAPSHOT)
    assert judge.validate_judgment(verdict(),packet)['status']=='unsupported'


def test_missing_input_snapshot_cannot_prove_justified_absence(materials):
    report={'generated_report':{'executive_summary':'The minimum separation is unknown.'}}
    packet=judge.prepare_packet(report,materials['questions'][0],materials)
    v={**verdict(),'status':'justified_abstention','report_quote':'The minimum separation is unknown.',
       'input_answerability':'unknown','input_id':None,'input_quote':''}
    assert judge.validate_judgment(v,packet)['status']=='requires_review'


def test_citation_presence_only_and_untrusted_numeric_judge_scores_are_rejected(materials):
    packet=judge.prepare_packet(REPORT,materials['questions'][0],materials,SNAPSHOT)
    v={**verdict(),'score':100}
    with pytest.raises(ValueError):judge.validate_judgment(v,packet)


def test_bounded_input_selection_never_claims_absence_in_omitted_material(materials):
    snapshot={'complete':True,'evidence':[
        {'chunk_id':f'E{i}','source_type':'official_factual','evidence_role':'case_factual',
         'excerpt':('Long factual context. '*200)+FACT} for i in range(6)]}
    packet=judge.prepare_packet(REPORT,materials['questions'][0],materials,snapshot)
    assert len(packet['input_fields'])==3
    assert not packet['input_snapshot_complete']
    assert packet['report_fields']==judge.report_fields(REPORT)
    assert all(len(row['text'])<=1600 for row in packet['input_fields'].values())
    v={**verdict(),'report_quote':'INPUT:1'}
    with pytest.raises(ValueError):judge.validate_judgment(v,packet)


def test_default_provider_adapter_returns_strict_fields_not_computed_points(monkeypatch,materials):
    async def evaluate(settings,system,payload,validator):
        return validator(verdict())
    monkeypatch.setattr(judge,'validated_evaluation',evaluate)
    packet=judge.prepare_packet(REPORT,materials['questions'][0],materials,SNAPSHOT)
    value=asyncio.run(judge.judge_question(settings(),packet))
    assert 'points' not in value
    assert judge.validate_judgment(value,packet)['points']==1


def test_incremental_results_survive_failure_and_resume_without_rejudging_valid_rows(materials):
    materials['questions']*=3
    materials['questions']=[{**q,'question_id':f'Q{i}'} for i,q in enumerate(materials['questions'],1)]
    persisted=[];calls=[]
    async def failing(settings,packet):
        calls.append(packet['question_id'])
        if packet['question_id']=='Q2':raise httpx.ReadTimeout('SECRET_KEY_MUST_NOT_LEAK')
        return verdict()
    report=deepcopy(REPORT)
    first=asyncio.run(judge.evaluate_report(report,materials,settings(),run_id='RUN',snapshot=SNAPSHOT,
                                          judge=failing,save=lambda x:persisted.append(x)))
    assert calls==['Q1','Q2'] and first['score'] is None and first['questions_pending']==1
    assert any(len(s['judgments'])==1 for s in persisted)
    assert any(e['event']=='waiting_for_model_response' for s in persisted for e in s['events'])
    assert 'SECRET_KEY_MUST_NOT_LEAK' not in json.dumps(first)
    calls.clear()
    async def good(settings,packet):calls.append(packet['question_id']);return verdict()
    final=asyncio.run(judge.evaluate_report(report,materials,settings(),run_id='RUN',snapshot=SNAPSHOT,prior=first,judge=good))
    assert calls==['Q2','Q3'] and final['score']==100 and report==REPORT


def test_invalid_quote_does_not_prevent_later_questions(materials):
    materials['questions']=[{**materials['questions'][0], 'question_id':f'Q{i}'} for i in range(1,4)]
    calls=[]
    async def judge_one(settings, packet):
        calls.append(packet['question_id'])
        if packet['question_id']=='Q2':
            raise judge.EvaluationError('invalid_reference_quote_or_page')
        return verdict()
    result=asyncio.run(judge.evaluate_report(REPORT,materials,settings(),run_id='RUN',snapshot=SNAPSHOT,judge=judge_one))
    assert calls==['Q1','Q2','Q3']
    assert result['questions_failed']==1 and result['questions_pending']==0
    assert result['status']=='incomplete' and result['score'] is None


def test_model_and_report_changes_invalidate_cached_judgments(materials):
    calls=[]
    async def good(s,p):calls.append(p['question_id']);return verdict()
    initial=asyncio.run(judge.evaluate_report(REPORT,materials,settings(),run_id='RUN',snapshot=SNAPSHOT,judge=good))
    calls.clear()
    result=asyncio.run(judge.evaluate_report(REPORT,materials,replace(settings(),api_model='different-model'),
        run_id='RUN',snapshot=SNAPSHOT,prior=initial,judge=good))
    assert calls==['Q1'] and result['cache_key']!=initial['cache_key']
    altered={**REPORT,'unrelated_report_metadata':'changed'}
    assert judge.evaluation_identity(altered,materials,settings(),run_id='RUN')['report_sha256']!=initial['report_sha256']


def test_cancellation_persists_interruption_without_a_score(materials):
    saved=[]
    async def cancel(s,p):raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(judge.evaluate_report(REPORT,materials,settings(),run_id='RUN',snapshot=SNAPSHOT,judge=cancel,save=saved.append))
    assert saved[-1]['status']=='interrupted' and saved[-1]['score'] is None


def test_reference_change_during_evaluation_refuses_a_score(materials):
    from pathlib import Path
    saved=[]
    async def change(s,p):Path(materials['reference_path']).write_bytes(b'changed');return verdict()
    with pytest.raises(ValueError):
        asyncio.run(judge.evaluate_report(REPORT,materials,settings(),run_id='RUN',snapshot=SNAPSHOT,judge=change,save=saved.append))
    assert saved[-1]['status']=='failed' and saved[-1]['score'] is None


def test_api_saves_results_and_get_never_calls_judge(tmp_path,monkeypatch,materials):
    repo=SQLiteRepository(tmp_path/'jobs.sqlite3')
    repo.create_case('CASE',{'narrative':'Reported event','category':'runway_incursion'})
    repo.create_run('RUN','CASE','runway_incursion_investigation','mock')
    repo.patch_run('RUN',{'status':'completed','result':deepcopy(REPORT),'evaluation_input_snapshot':SNAPSHOT})
    repo.update_case('CASE',{'status':'analysis_complete'})
    calls=[]
    async def good(s,p):calls.append(p['question_id']);return verdict()
    monkeypatch.setattr(evaluation_routes,'case_materials',lambda case:materials)
    app=FastAPI();app.include_router(evaluation_routes.build_router(lambda:repo,settings))
    app.dependency_overrides[evaluation_routes.get_question_judge]=lambda:good
    path='/api/v1/cases/CASE/runs/RUN/report-evaluation'
    with TestClient(app) as client:
        assert client.get(path).json()['status']=='not_requested' and not calls
        assert client.post(path).status_code==202
        result=client.get(path).json();assert result['status']=='completed' and result['score']==100
        assert client.post(path).status_code==202 and calls==['Q1']
    repo.close()


def test_restart_recovery_retains_partial_judgments(tmp_path,monkeypatch):
    path=tmp_path/'recovery.sqlite3';repo=SQLiteRepository(path)
    repo.create_case('CASE',{'narrative':'Reported event','category':'runway_incursion'})
    repo.create_run('RUN','CASE','runway_incursion_investigation','mock')
    repo.patch_run('RUN',{'status':'completed','result':REPORT,'question_evaluation':{
        'status':'running','process_id':'previous-process','judgments':[{'question_id':'Q1','status':'correct','points':1}]}})
    repo.close();repo=SQLiteRepository(path)
    monkeypatch.setattr(evaluation_routes,'PROCESS_ID','new-process')
    recovered=evaluation_routes._recover(repo,repo.get_run('RUN'))
    assert recovered['status']=='interrupted' and len(recovered['judgments'])==1
    assert repo.get_run('RUN')['result']==REPORT
    repo.close()


def test_unexpected_background_exit_does_not_terminate_server(tmp_path,monkeypatch,materials):
    repo=SQLiteRepository(tmp_path/'fail.sqlite3')
    repo.create_case('CASE',{'narrative':'Reported event','category':'runway_incursion'})
    repo.create_run('RUN','CASE','runway_incursion_investigation','mock')
    repo.patch_run('RUN',{'status':'completed','result':REPORT})
    repo.update_case('CASE',{'status':'analysis_complete'})
    async def bad(s,p):raise SystemExit('must be contained')
    monkeypatch.setattr(evaluation_routes,'case_materials',lambda c:materials)
    app=FastAPI();app.include_router(evaluation_routes.build_router(lambda:repo,settings))
    app.dependency_overrides[evaluation_routes.get_question_judge]=lambda:bad
    path='/api/v1/cases/CASE/runs/RUN/report-evaluation'
    with TestClient(app) as client:
        assert client.post(path).status_code==202
        result=client.get(path);assert result.status_code==200 and result.json()['status']=='failed'
    repo.close()


def test_workflow_checkpoint_captures_generation_inputs_without_gold_or_model_calls(tmp_path,monkeypatch):
    from blackwing import api
    repo=SQLiteRepository(tmp_path/'snapshot.sqlite3')
    case=repo.create_case('CASE',{'narrative':'A reported runway event.','category':'runway_incursion'})
    repo.create_run('RUN','CASE','runway_incursion_investigation','mock')
    async def runner(narrative, settings, **kwargs):
        kwargs['on_update']({'evidence':SNAPSHOT['evidence'],'agent_evidence':{},'outputs':{},'trace':[]})
        return {**deepcopy(REPORT),'trace':[],'model':'mock'}
    monkeypatch.setattr(api,'analyze_incident',runner)
    asyncio.run(api._execute_run('RUN',case,[],settings(),repo,runner,use_agent_retrieval=False))
    snapshot=repo.get_run('RUN')['evaluation_input_snapshot']
    assert snapshot['complete'] is True and snapshot['evidence']==SNAPSHOT['evidence']
    assert not any('gold' in str(row.get('source_type','')).casefold() for row in snapshot['evidence'])
    repo.close()
