import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from blackwing import retrieval_evaluation_routes as routes


def test_route_recomputes_scoped_recall_from_approved_qrels_and_saved_rankings(tmp_path, monkeypatch):
    qrels = {
        'unit': 'document', 'index_signature': 'sig', 'scope_id': 'qrels-test',
        'scope_description': 'One reviewed query in a bounded test corpus.',
        'full_corpus_relevance_complete': True,
        'approval': {'status': 'APPROVED'}, 'eligible_ids': ['A', 'B', 'C'],
        'questions': [{'question_id': 'Q1', 'query': 'query', 'complete_relevance': True,
                       'relevant_ids': ['A', 'C']}],
    }
    rankings = {
        'unit': 'document', 'index_signature': 'sig', 'scope_id': 'qrels-test',
        'queries': [{'question_id': 'Q1', 'method': 'bm25', 'ranked_ids': ['A', 'B', 'C']}],
    }
    qrels_path, rankings_path = tmp_path / 'qrels.json', tmp_path / 'rankings.json'
    qrels_path.write_text(json.dumps(qrels), encoding='utf-8')
    rankings_path.write_text(json.dumps(rankings), encoding='utf-8')
    index_dir = tmp_path / 'index'
    index_dir.mkdir()
    (index_dir / 'manifest.json').write_text(json.dumps({'signature': 'sig'}), encoding='utf-8')
    monkeypatch.setattr(routes, 'QRELS_PATH', qrels_path)
    monkeypatch.setattr(routes, 'RANKINGS_PATH', rankings_path)

    class Settings:
        rag_index_dir = str(index_dir)

    app = FastAPI()
    app.include_router(routes.build_router(lambda: Settings()))
    result = TestClient(app).get('/api/v1/evaluations/retrieval').json()

    assert result['status'] == 'completed'
    assert result['summary'][0]['recall@5'] == 1.0
    assert result['summary'][0]['recall@10'] == 1.0


def test_route_marks_benchmark_stale_for_different_configured_index(tmp_path, monkeypatch):
    qrels_path, rankings_path = tmp_path / 'qrels.json', tmp_path / 'rankings.json'
    qrels_path.write_text(json.dumps({'index_signature': 'old'}), encoding='utf-8')
    rankings_path.write_text('{}', encoding='utf-8')
    index_dir = tmp_path / 'index'
    index_dir.mkdir()
    (index_dir / 'manifest.json').write_text(json.dumps({'signature': 'new'}), encoding='utf-8')
    monkeypatch.setattr(routes, 'QRELS_PATH', qrels_path)
    monkeypatch.setattr(routes, 'RANKINGS_PATH', rankings_path)

    class Settings:
        rag_index_dir = str(index_dir)

    app = FastAPI()
    app.include_router(routes.build_router(lambda: Settings()))
    result = TestClient(app).get('/api/v1/evaluations/retrieval').json()
    assert result['status'] == 'stale'
