from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from blackwing import api, copilot


SCRIPT = """from blackwing.evaluation_ui import show_run_trace
show_run_trace({'run_id': 'RUN', 'case_id': 'CASE', 'status': 'completed', 'model': 'test-model'},
               {'limitations': ['Human review is required.']})
"""


def test_run_trace_renders_structured_timeline_without_raw_json(monkeypatch):
    monkeypatch.setattr(copilot, 'get_trace', lambda run_id: {
        'run_id': run_id,
        'events': [{'sequence': 1, 'timestamp': '2026-01-01T00:00:00+00:00',
                    'step': 'investigation', 'agent': 'Investigation', 'agent_key': 'investigation',
                    'status': 'agent_completed', 'duration_ms': 123.4}],
    })
    monkeypatch.setattr(copilot, 'get_audit', lambda run_id: {
        'run_id': run_id, 'status': 'completed', 'model': 'test-model',
        'execution_steps': 8, 'operations_metrics': {
            'latency_ms': 1000, 'tool_calls': 2, 'prompt_tokens': 100, 'completion_tokens': 50,
        },
        'events': [{'event_id': 'tool-1', 'timestamp': '2026-01-01T00:00:01+00:00',
                    'step': 'retrieve_evidence', 'tool': 'retrieve_evidence', 'status': 'complete',
                    'duration_ms': 25, 'result_summary': {'count': 3}}],
        'plan': {'agents': ['investigation']},
    })
    monkeypatch.setattr(copilot, 'get_final_report_evaluation', lambda *args: {
        'status': 'completed', 'sections_total': 1,
        'sections': {'overview': {'llm_judge': {'score': 80}, 'cosine': {'cosine_similarity': .8}}},
    })
    monkeypatch.setattr(copilot, 'get_report_evaluation', lambda *args: {
        'status': 'not_requested', 'questions_total': 0,
    })
    monkeypatch.setattr(copilot, 'get_retrieval_evaluation', lambda: {
        'status': 'completed', 'unit': 'document', 'scope_id': 'approved-test-scope',
        'scope_description': 'Five queries over 50 reviewed sources.',
        'summary': [{'method': 'bm25', 'queries_total': 5, 'queries_measured': 5,
                     'recall@5': 0.4, 'recall@10': 0.5333}],
        'queries': [{'question_id': 'Q1', 'method': 'bm25', 'status': 'measured',
                     'relevant_count': 3, 'relevant_retrieved@5': 2, 'recall@5': 2/3,
                     'relevant_retrieved@10': 2, 'recall@10': 2/3}],
    })

    at = AppTest.from_string(SCRIPT).run()

    assert not at.exception
    assert any(metric.label == 'Tool calls' and metric.value == '2' for metric in at.metric)
    assert len(at.dataframe) >= 2  # Timeline and evaluation activity tables.
    assert any('scores stay in the Investigation report' in caption.value for caption in at.caption)
    assert any(metric.label == 'BM25 Recall@5' and metric.value == '40.0%' for metric in at.metric)
    assert any(metric.label == 'BM25 Recall@10' and metric.value == '53.3%' for metric in at.metric)


def test_swagger_documents_trace_audit_and_operations_metrics_contracts():
    schema = api.app.openapi()
    paths = schema['paths']

    assert '/api/v1/runs/{run_id}/trace' in paths
    assert '/api/v1/runs/{run_id}/audit' in paths
    assert '/api/v1/metrics/operations' in paths
    assert '/api/v1/evaluations/retrieval' in paths
    assert paths['/api/v1/runs/{run_id}/trace']['get']['summary'] == 'Get a chronological run trace'
    audit_response = paths['/api/v1/runs/{run_id}/audit']['get']['responses']['200']['content']['application/json']['schema']
    assert audit_response['$ref'].endswith('/RunAuditResponse')
    metric_response = paths['/api/v1/metrics/operations']['get']['responses']['200']['content']['application/json']['schema']
    assert metric_response['$ref'].endswith('/OperationsMetricsResponse')


def test_running_run_does_not_poll_report_evaluation_endpoints(monkeypatch):
    monkeypatch.setattr(copilot, 'get_trace', lambda run_id: {
        'run_id': run_id,
        'events': [{'sequence': 1, 'step': 'investigation', 'status': 'running'}],
    })
    monkeypatch.setattr(copilot, 'get_audit', lambda run_id: {
        'run_id': run_id, 'status': 'running', 'operations_metrics': {}, 'events': [],
    })

    def should_not_poll(*args):
        raise AssertionError('Report evaluation endpoint must not be polled before completion.')

    monkeypatch.setattr(copilot, 'get_final_report_evaluation', should_not_poll)
    monkeypatch.setattr(copilot, 'get_report_evaluation', should_not_poll)
    monkeypatch.setattr(copilot, 'get_retrieval_evaluation', lambda: {'status': 'not_configured'})

    at = AppTest.from_string(SCRIPT.replace("'completed'", "'running'")).run()

    assert not at.exception
    assert any(metric.label == 'Run status' and metric.value == 'Running' for metric in at.metric)
    assert len(at.dataframe) >= 2  # Timeline and waiting evaluation activity.


def test_audit_endpoint_returns_documented_operations_fields():
    class Repository:
        def get_run(self, run_id):
            return {
                'run_id': run_id, 'status': 'completed', 'workflow': 'runway_incursion_investigation',
                'model': 'test-model', 'audit_events': [{
                    'step': 'retrieve_evidence', 'status': 'complete',
                    'retrieved_document_count': 3, 'timestamp': '2026-01-01T00:00:00+00:00',
                }],
                'operations_metrics': {'latency_ms': 1200, 'tool_calls': 1,
                                       'prompt_tokens': 40, 'completion_tokens': 10},
            }

    api.app.dependency_overrides[api.get_repository] = lambda: Repository()
    try:
        response = TestClient(api.app).get('/api/v1/runs/RUN/audit')
    finally:
        api.app.dependency_overrides.pop(api.get_repository, None)

    assert response.status_code == 200
    body = response.json()
    assert body['operations_metrics']['latency_ms'] == 1200
    assert body['events'][0]['retrieved_document_count'] == 3
