"""Offline UI/API checks: no implicit inference and no invented final score."""
from copy import deepcopy
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import pytest
from streamlit.testing.v1 import AppTest

from blackwing import copilot, evaluation_routes
from blackwing.evaluation.materials import MaterialsUnavailable
from blackwing.repository import SQLiteRepository
from test_agentic_local import settings
from test_question_report_evaluation import FACT, REPORT, materials, verdict


SCRIPT = """from blackwing.copilot_ui import show_gold_comparison
show_gold_comparison({'incident': {'case_id': 'CASE'}, 'report_run_id': 'RUN'})
"""


def snapshot(status='not_requested', **extra):
    return {'status': status, 'score': None, 'reference_id': 'REF',
            'reference_case': True, 'questions_total': 2, **extra}


def forbid(*args, **kwargs):
    raise AssertionError('Model work must require an explicit comparison click.')


def test_opening_report_only_reads_question_status(monkeypatch):
    monkeypatch.setattr(copilot, 'get_report_evaluation', lambda *args: snapshot())
    monkeypatch.setattr(copilot, 'start_report_evaluation', forbid)
    monkeypatch.setattr(copilot, 'compare_with_gold', forbid)
    at = AppTest.from_string(SCRIPT).run()
    assert not at.exception
    assert at.metric[0].label == 'Gold comparison score'
    assert at.metric[0].value == 'Not assessed'
    assert any(b.label == 'Compare with gold report' for b in at.button)
    assert any('approved reference' in c.value for c in at.caption)
    assert not any('Question-based comparison' in c.value for c in at.caption)


def test_explicit_click_queues_once_and_refresh_only_reads_saved_result(monkeypatch):
    state = snapshot()
    posts = []
    monkeypatch.setattr(copilot, 'get_report_evaluation', lambda *args: deepcopy(state))
    def start(case_id, run_id):
        posts.append((case_id, run_id))
        state.update(status='queued')
        return deepcopy(state)
    monkeypatch.setattr(copilot, 'start_report_evaluation', start)
    at = AppTest.from_string(SCRIPT).run()
    at.button(key='compare-gold-RUN').click().run()
    assert not at.exception and posts == [('CASE', 'RUN')]
    assert not any(b.label == 'Compare with gold report' for b in at.button)
    state.update(status='completed', score=75, questions_completed=2, questions_scored=2,
                 judgments=[{'question_id': 'Q1', **verdict(), 'points': 1}],
                 improvements=[{'question_id':'Q2', 'report_field':None,
                                'issue':'An answer is missing.', 'action':'Review available supporting evidence.'}])
    at.button(key='refresh-gold-RUN').click().run()
    assert not at.exception and posts == [('CASE', 'RUN')]
    assert at.metric[0].value == '75/100'
    assert at.dataframe[0].value.iloc[0]['report_quote'] == FACT
    assert any(e.label == 'Improvements to review' for e in at.expander)


def test_active_comparison_rereads_status_and_displays_terminal_result(monkeypatch):
    state = snapshot('running', questions_completed=1, events=[
        {'event':'waiting_for_model_response', 'question_id':'Q2'}])
    reads=[]
    def read(*args):
        reads.append(args)
        return deepcopy(state)
    monkeypatch.setattr(copilot, 'get_report_evaluation', read)
    monkeypatch.setattr(copilot, 'start_report_evaluation', forbid)
    at=AppTest.from_string(SCRIPT).run()
    assert not at.exception
    assert any('Q2' in item.value and 'automatically' in item.value for item in at.info)
    state.update(status='incomplete', questions_completed=1, questions_failed=1)
    at.run()
    assert not at.exception and len(reads)>=2
    assert any('incomplete' in item.value for item in at.caption)
    assert at.metric[0].value=='Not assessed'
    assert any(b.label=='Retry unvalidated comparisons' for b in at.button)


@pytest.mark.parametrize('status', ['queued', 'running', 'incomplete', 'failed', 'interrupted', 'requires_review'])
def test_nonfinal_states_never_display_a_numeric_final_score(monkeypatch, status):
    monkeypatch.setattr(copilot, 'get_report_evaluation', lambda *args: snapshot(status, score=100))
    monkeypatch.setattr(copilot, 'start_report_evaluation', forbid)
    at = AppTest.from_string(SCRIPT).run()
    assert not at.exception and at.metric[0].value == 'Not assessed'
    if status in {'queued', 'running'}:
        assert not any('Compare' in b.label or 'Retry' in b.label for b in at.button)


def test_justified_abstention_only_result_is_not_invented_as_zero_or_100(monkeypatch):
    monkeypatch.setattr(copilot, 'get_report_evaluation', lambda *args: snapshot(
        'not_assessable', questions_completed=2, questions_excluded=2))
    at = AppTest.from_string(SCRIPT).run()
    assert not at.exception and at.metric[0].value == 'Not assessable'
    assert any('Justified abstention' in i.value for i in at.info)


def test_post_timeout_is_sanitized_and_keeps_refresh_available(monkeypatch):
    monkeypatch.setattr(copilot, 'get_report_evaluation', lambda *args: snapshot())
    def timeout(*args):
        raise httpx.ReadTimeout('SECRET PROVIDER DETAILS')
    monkeypatch.setattr(copilot, 'start_report_evaluation', timeout)
    at = AppTest.from_string(SCRIPT).run()
    at.button(key='compare-gold-RUN').click().run()
    assert not at.exception and at.error
    assert 'SECRET' not in at.error[0].value
    assert 'refresh status before retrying' in at.error[0].value
    assert at.button(key='refresh-gold-RUN')


def test_unverified_materials_do_not_fall_back_to_a_misleading_score(monkeypatch):
    monkeypatch.setattr(copilot, 'get_report_evaluation', lambda *args: snapshot(
        'unavailable', legacy_fallback=False, reason='Matching reference hash could not be verified.'))
    monkeypatch.setattr(copilot, 'get_gold_comparison', forbid)
    at = AppTest.from_string(SCRIPT).run()
    assert not at.exception and at.metric[0].value == 'Unavailable' and not at.button


@pytest.fixture
def api_client(tmp_path):
    repo = SQLiteRepository(tmp_path/'ui.sqlite3')
    repo.create_case('CASE', {'narrative':'A reported event.', 'category':'runway_incursion'})
    repo.create_run('RUN', 'CASE', 'runway_incursion_investigation', 'mock')
    repo.patch_run('RUN', {'status':'completed', 'result':deepcopy(REPORT)})
    repo.update_case('CASE', {'status':'analysis_complete'})
    app = FastAPI()
    app.include_router(evaluation_routes.build_router(lambda:repo, settings))
    app.dependency_overrides[evaluation_routes.get_question_judge] = lambda:forbid
    with TestClient(app) as client:
        yield client
    repo.close()


def test_availability_exposes_counts_but_no_expected_answers_or_reference_text(api_client, monkeypatch, materials):
    monkeypatch.setattr(evaluation_routes, 'case_materials', lambda case:materials)
    response = api_client.get('/api/v1/cases/CASE/runs/RUN/report-evaluation')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'not_requested' and data['questions_total'] == 1
    assert data['human_review_required'] is True
    assert FACT not in response.text and 'expected_answer' not in response.text


def test_absent_approved_questions_preserve_explicit_legacy_scope(api_client, monkeypatch):
    def missing(case):
        raise MaterialsUnavailable('Approved questions are unavailable.', reference_case=True)
    monkeypatch.setattr(evaluation_routes, 'case_materials', missing)
    result = api_client.get('/api/v1/cases/CASE/runs/RUN/report-evaluation').json()
    assert result['legacy_fallback'] is True and result['reference_case'] is True
    assert result['score'] is None


def test_reference_error_is_sanitized_without_legacy_fallback(api_client, monkeypatch):
    def invalid(case):
        raise ValueError('SECRET absolute protected path')
    monkeypatch.setattr(evaluation_routes, 'case_materials', invalid)
    response = api_client.get('/api/v1/cases/CASE/runs/RUN/report-evaluation')
    assert response.json()['legacy_fallback'] is False
    assert 'SECRET' not in response.text and response.json()['score'] is None
