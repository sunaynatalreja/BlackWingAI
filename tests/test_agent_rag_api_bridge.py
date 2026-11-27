from __future__ import annotations

from typing import Any
from fastapi.testclient import TestClient

from blackwing.api import app, get_analysis_runner, get_repository, get_settings
from blackwing.config import Settings
from blackwing.repository import MemoryRepository
from blackwing.rag.engine import HybridIndex
from blackwing.rag.retriever import _compat_document
from blackwing.rag.tokenization import ApproximateTokenizer


def test_rag_packets_reach_all_seven_agents(monkeypatch):
    from blackwing import api
    monkeypatch.setenv("BLACKWING_AGENT_RAG", "1")
    monkeypatch.setenv("BLACKWING_AGENT_RAG_METHOD", "hybrid")
    repo = MemoryRepository()
    captured: dict[str, Any] = {}

    class BM25CompatibleRetriever:
        def search(self, query, documents, *, limit=5, method="hybrid", scope="all", **kwargs):
            index = HybridIndex.from_documents(
                [_compat_document(dict(doc)) for doc in documents],
                tokenizer=ApproximateTokenizer(), strategy="fixed")
            return index.search(query, top_k=limit, method="bm25", scope=scope,
                                case_id=kwargs.get("case_id"), jurisdiction=kwargs.get("jurisdiction"),
                                incident_date=kwargs.get("incident_date"),
                                document_types=kwargs.get("document_types"))

    monkeypatch.setattr(api, "retriever", BM25CompatibleRetriever())
    settings = Settings(host="127.0.0.1", port=8000, llm_provider="ollama",
                        ollama_base_url="http://localhost:11434", ollama_model="qwen3:8b",
                        ollama_timeout_seconds=1, ollama_temperature=0, ollama_think=False,
                        ollama_max_output_tokens=512, api_base_url="", api_model="", api_key="",
                        api_timeout_seconds=1, api_temperature=0, api_max_output_tokens=512,
                        ui_api_url="http://127.0.0.1:8000", ui_request_timeout_seconds=1,
                        rag_embedding_model="BAAI/bge-small-en-v1.5")

    async def fake_runner(narrative, settings, **kwargs):
        captured.update(kwargs)
        return {"case_id": kwargs["case_id"], "model": "mock-not-Qwen",
                "agent_outputs": {}, "trace": [], "status": "PRELIMINARY_REQUIRES_HUMAN_REVIEW"}

    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_analysis_runner] = lambda: fake_runner
    try:
        with TestClient(app) as client:
            d = client.post("/api/v1/knowledge/documents", json={
                "title": "Old runway entry procedures", "document_type": "procedure",
                "source_id": "US-PROC-TEST", "jurisdiction": "US",
                "effective_date": "2020-01-01", "excerpt": "ATC clearance rule note",
                "content": "Runway holding position clearance and readback movement area requirements."})
            assert d.status_code == 201
            c = client.post("/api/v1/cases", json={
                "category": "runway_incursion",
                "narrative": "A vehicle moved across runway holding position after an ATC clearance misunderstanding.",
                "incident_date": "2024-01-01", "jurisdiction": "US",
                "evidence": [{"title": "Observer note", "document_type": "observer_report",
                              "excerpt": "Observer saw a ground vehicle close to runway position."}]
            })
            assert c.status_code == 201
            case_id = c.json()["case_id"]
            r = client.post(f"/api/v1/cases/{case_id}/runs", json={"include_retrieval": True})
            assert r.status_code == 202
            assert client.get(f"/api/v1/runs/{r.json()['run_id']}").json()["status"] == "completed"
    finally:
        app.dependency_overrides.clear()
    packets = captured["agent_evidence"]
    assert len(packets) == 7
    assert packets["investigation"][0]["source_type"] == "user_submitted"
    assert all(item["source_type"] != "official_final" for item in packets["investigation"])
    assert "US-PROC-TEST" in {item["source_id"] for item in packets["compliance"]}
    assert len(packets["compliance"]) <= 3
