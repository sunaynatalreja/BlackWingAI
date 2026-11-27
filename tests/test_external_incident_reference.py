"""An external incident identifier must be used for matching factual index records.

This does not change the application's internally generated UUID or UI contract.
"""

import asyncio
from types import SimpleNamespace

from blackwing import api
from blackwing.schemas import CaseCreate


class _FakeRetriever:
    def __init__(self):
        self.direct_case_ids = []

    def search(self, query, documents, **kwargs):
        if kwargs.get("scope") == "direct":
            self.direct_case_ids.append(kwargs.get("case_id"))
        return {"results": []}


class _FakeRepository:
    def __init__(self):
        self.updated = []
    def patch_run(self, run_id, fields):
        self.updated.append((run_id, fields))
    def update_case(self, case_id, fields):
        pass
    def list_documents(self):
        return []


def test_explicit_external_incident_id_routes_to_index(monkeypatch):
    payload = CaseCreate.model_validate({
        "source_incident_id": "EXAMPLE-CASE-2023-99",
        "narrative": "This is a deliberately synthetic runway-incursion incident narrative.",
        "incident_date": "2023-02-04",
        "jurisdiction": "US",
    })
    assert payload.source_incident_id == "EXAMPLE-CASE-2023-99"

    fake_retriever = _FakeRetriever()
    monkeypatch.setattr(api, "retriever", fake_retriever)
    monkeypatch.setenv("BLACKWING_AGENT_RAG", "1")
    repository = _FakeRepository()
    case = {
        "case_id": "internal-random-uuid",
        "source_incident_id": "EXAMPLE-CASE-2023-99",
        "narrative": payload.narrative,
        "jurisdiction": "US",
        "incident_date": "2023-02-04",
        "evidence": [],
    }
    async def runner(narrative, settings, **kwargs):
        assert kwargs["case_id"] == "internal-random-uuid"
        assert len(kwargs["agent_evidence"]) == 7
        return {"trace": [], "model": "test", "agent_outputs": {}}
    asyncio.run(api._execute_run(
        "test-run", case, [],
        SimpleNamespace(rag_embedding_model="BAAI/bge-small-en-v1.5", model_name="test"),
        repository, runner,
    ))
    assert fake_retriever.direct_case_ids == ["EXAMPLE-CASE-2023-99"]
    assert any(p.get("status") == "completed" for _, p in repository.updated)
