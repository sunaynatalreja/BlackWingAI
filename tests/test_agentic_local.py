"""Real tools, graph, SQLite and HTTP contracts; only model/provider IO is mocked."""
import asyncio
from collections import Counter
from dataclasses import replace
import json
import subprocess
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from blackwing import api, service
from blackwing.agents import runtime
from blackwing.agents.base import AgentExecutionError
from blackwing.config import Settings
from blackwing.planning import plan_investigation, correction_target
from blackwing.repository import SQLiteRepository
from blackwing.tools import tool_audit, retrieve_evidence, assess_reference_risk


def settings():
    return Settings(host="127.0.0.1", port=8001, llm_provider="ollama",
                    ollama_base_url="http://localhost:11434", ollama_model="qwen3:8b",
                    ollama_timeout_seconds=1, ollama_temperature=0, ollama_think=False,
                    ollama_max_output_tokens=512, api_base_url="https://iit.invalid/v1",
                    api_model="qwen3-8b", api_key="", api_timeout_seconds=1,
                    api_temperature=0, api_max_output_tokens=512, ui_api_url="http://test",
                    ui_request_timeout_seconds=1, rag_embedding_model="unused")


DIRECT = {"chunk_id": "CUR:1", "document_id": "CUR", "source_id": "CUR",
          "incident_id": "SYNTH-1", "source_type": "official_factual", "document_type": "incident_report",
          "evidence_role": "case_factual", "excerpt": "Aircraft crossed runway during taxi clearance."}
HISTORICAL = {**DIRECT, "chunk_id": "OLD:1", "source_id": "OLD", "incident_id": "OLD-1",
              "evidence_role": "historical_analogy", "source_type": "official_final"}


def mock_agents(monkeypatch, *, correction=False, impossible=False, fail_correction=False):
    calls = []
    async def invoke(spec, **kwargs):
        calls.append(spec.key)
        if spec.key == "investigation":
            return {"incident_summary": "Synthetic conflict", "documented_facts": [] if impossible else [
                {"text": "Aircraft crossed runway", "evidence_ids": ["CUR:1"], "source_ids": [], "evidence_state": "documented_fact"}]}
        if spec.key == "risk":
            return {"hazards": [{"hazard_id": "H1", "description": "Runway conflict", "evidence_ids": [] if impossible else ["CUR:1"],
                                 "severity": 2, "likelihood": "C", "severity_basis": "Synthetic severity basis", "likelihood_basis": "Synthetic likelihood basis"}]}
        if spec.key == "mitigation":
            count = calls.count("mitigation")
            if count == 2 and fail_correction:
                raise ValueError("Mock unavailable")
            return {"recommendations": [] if correction and count == 1 else [
                {"action": "Review clearance coordination", "hazard_ids": ["H1"]}]}
        if spec.key == "critic":
            return {"issues": []}
        if spec.key == "verification":
            return {"human_approval_required": True}
        raise AssertionError(spec.key)
    monkeypatch.setattr(service, "invoke_agent", invoke)
    return calls


def investigate(monkeypatch, **flags):
    calls = mock_agents(monkeypatch, **flags)
    packets = {spec.key: ([] if flags.get("impossible") else [DIRECT]) for spec in service.AGENT_SPECS}
    packets["similar_case"] = [HISTORICAL]
    packets["compliance"] = []
    result = asyncio.run(service.analyze_incident("Synthetic taxi clearance runway conflict", settings(),
                        case_id="case-1", source_incident_id="SYNTH-1", agent_evidence=packets))
    return result, calls


def outcome(result):
    return next(e["detail"] for e in result["trace"] if e["step"] == "investigation_outcome")


def test_critic_targeted_correction_succeeds_once(monkeypatch):
    result, calls = investigate(monkeypatch, correction=True)
    assert Counter(calls) == {"investigation": 1, "risk": 1, "mitigation": 2, "critic": 1, "verification": 1}
    assert outcome(result)["execution_steps"] == 10
    assert outcome(result)["correction_attempts"] == 1
    assert next(e for e in result["trace"] if e["step"] == "correction_outcome")["status"] == "corrected"
    assert not any(i["issue_type"] == "no_grounded_mitigations" for i in result["agent_outputs"]["critic"]["issues"])
    assert result["agent_outputs"]["risk"]["hazards"][0]["risk_index"] == "C2"
    assert result["agent_outputs"]["mitigation"]["recommendations"][0]["hazard_ids"] == ["H1"]
    assert result["generated_report"]["human_approval_required"] is True


def test_impossible_correction_terminates_with_unknowns(monkeypatch):
    result, calls = investigate(monkeypatch, correction=True, impossible=True)
    assert calls.count("mitigation") == 1
    assert outcome(result)["execution_steps"] == 9
    assert outcome(result)["correction_attempts"] == 0
    risk = result["agent_outputs"]["risk"]["hazards"][0]
    assert risk["risk_index"] is None and risk["evidence_ids"] == []
    assert result["agent_outputs"]["critic"]["issues"]
    assert result["agent_outputs"]["investigation"]["documented_facts"] == []


def test_failed_correction_retains_original_and_stops(monkeypatch):
    result, calls = investigate(monkeypatch, correction=True, fail_correction=True)
    assert calls.count("mitigation") == 2
    assert outcome(result)["execution_steps"] == 10
    assert result["agent_outputs"]["mitigation"]["recommendations"] == []
    assert any(e["status"] == "failed_requires_human_review" for e in result["trace"])


def test_execution_budget_guard_prevents_extra_agent(monkeypatch):
    monkeypatch.setattr(service, "MAX_EXECUTION_STEPS", 6)
    with pytest.raises(AgentExecutionError) as exc:
        investigate(monkeypatch)
    assert exc.value.code == "execution_step_limit"


def test_planner_routes_rules_and_evidence_without_airport_logic():
    for airport in ("JFK", "Austin", "Goa", "Delhi", "Synthetic"):
        state = {"narrative": airport, "agent_evidence": {"compliance": []}}
        plan = plan_investigation(state)
        assert "unassessed" in plan["decisions"]["compliance"]
        state["agent_evidence"]["compliance"] = [{"source_type": "regulation"}]
        assert plan_investigation(state)["decisions"]["compliance"] == "compare authoritative rules"
        assert plan["max_execution_steps"] == 10 and plan["max_corrections"] == 1
    assert correction_target({"correction_attempts": 1}) is None


def test_tool_failure_is_real_and_does_not_log_secret():
    events = []
    called = []
    def fail():
        called.append(True)
        raise RuntimeError("APIKEY-secret")
    with tool_audit("case", events.append):
        with pytest.raises(RuntimeError):
            retrieve_evidence(fail, input_identifiers=["index"])
        scored = assess_reference_risk()
    assert called == [True]
    assert events[0]["status"] == "failed" and events[0]["duration_ms"] >= 0
    assert "APIKEY" not in json.dumps(events)
    assert events[1]["status"] == "complete" and scored["risk_index"] is None


class RetrievalFixture:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
    def search(self, query, documents, **kwargs):
        self.calls.append(kwargs.get("scope", "all"))
        if self.fail:
            raise OSError("Synthetic index unavailable")
        rows = {"direct": [DIRECT], "historical": [HISTORICAL], "guidance": [], "regulatory": []}
        return {"results": rows.get(kwargs.get("scope"), [DIRECT]), "retrieval_method": "mock-index", "limitations": []}


@pytest.mark.parametrize("retrieval_failure", [False, True])
def test_api_real_three_tools_and_failure_recovery_restart(monkeypatch, tmp_path, retrieval_failure):
    calls = mock_agents(monkeypatch, impossible=retrieval_failure)
    index = RetrievalFixture(fail=retrieval_failure)
    monkeypatch.setattr(api, "retriever", index)
    monkeypatch.setenv("BLACKWING_AGENT_RAG", "1")
    repo = SQLiteRepository(tmp_path / "cases.sqlite3")
    api.app.dependency_overrides[api.get_repository] = lambda: repo
    api.app.dependency_overrides[api.get_settings] = settings
    try:
        with TestClient(api.app) as client:
            case = client.post("/api/v1/cases", json={"source_incident_id": "SYNTH-1", "narrative": "Synthetic taxi clearance runway conflict"}).json()
            case_id = case["case_id"]
            response = client.post(f"/api/v1/cases/{case_id}/runs", json={})
            assert response.status_code == 202
            run_id = response.json()["run_id"]
            assert client.get(f"/api/v1/runs/{run_id}").json()["status"] == "completed"
            trace = client.get(f"/api/v1/runs/{run_id}/trace").json()["events"]
            audits = client.get(f"/api/v1/runs/{run_id}/audit").json()
            tools = [e for e in audits["events"] if e.get("tool")]
            assert {e["tool"] for e in tools} == {"retrieve_evidence", "find_similar_incidents", "assess_reference_risk"}
            assert all(e["case_id"] == case_id and e["duration_ms"] >= 0 for e in tools)
            assert all("input_identifiers" in e and "result_summary" in e for e in tools)
            assert len([e for e in tools if e["tool"] == "retrieve_evidence"]) == 2
            assert len(index.calls) == (2 if retrieval_failure else 4)
            assert all(e["status"] == ("failed" if retrieval_failure else "complete") for e in tools if e["tool"] == "retrieve_evidence")
            assert {e["invocation_id"] for e in trace if e.get("tool")} == {e["invocation_id"] for e in tools}
            report = client.get(f"/api/v1/cases/{case_id}/report")
            assert report.status_code == 200, report.text
            assert report.json()["is_final"] is False
            for fmt in ("json", "md"):
                assert client.get(f"/api/v1/cases/{case_id}/report/export?format={fmt}").status_code == 200
            review = client.post(f"/api/v1/cases/{case_id}/reviews", json={"reviewer_id": "test-human", "decision": "request_changes", "rationale": "Check primary sources"})
            assert review.status_code == 201
            saved = repo.get_run(run_id)
        repo.close()
        repo = SQLiteRepository(tmp_path / "cases.sqlite3")
        assert repo.get_run(run_id)["result"] == saved["result"]
        assert repo.get_run(run_id)["audit_events"] == saved["audit_events"]
        assert repo.get_run(run_id)["report_references"]["run_id"] == run_id
        assert repo.list_reviews(case_id)[0]["decision"] == "request_changes"
    finally:
        api.app.dependency_overrides.clear()
        repo.close()


def test_sqlite_new_process_restart_and_interrupted_recovery(tmp_path):
    db = tmp_path / "restart.sqlite3"
    repo = SQLiteRepository(db)
    repo.create_case("case", {"narrative": "Synthetic incident", "category": "runway_incursion"})
    repo.add_evidence("case", {"evidence_id": "E1", "excerpt": "User report"})
    repo.add_document("D1", {"title": "User doc", "document_type": "user_report", "content": "Synthetic content"})
    repo.create_run("done", "case", "workflow", "mock")
    repo.patch_run("done", {"status": "completed", "result": {"agent_outputs": {"risk": {"hazards": []}}, "generated_report": {"reference": "R1"}}})
    repo.create_run("interrupted", "case", "workflow", "mock")
    repo.patch_run("interrupted", {"status": "running", "partial_outputs": {"investigation": {"unknowns": ["missing"]}}})
    repo.close()
    program = """import sys,json
from blackwing.repository import SQLiteRepository
r=SQLiteRepository(sys.argv[1])
assert r.get_run('done')['result']['generated_report']['reference']=='R1'
assert r.get_case('case')['evidence'][0]['evidence_id']=='E1'
assert r.get_document('D1')['content']=='Synthetic content'
assert r.get_run('interrupted')['error_code']=='interrupted'
assert r.get_run('interrupted')['partial_outputs']['investigation']['unknowns']==['missing']
r.create_run('recovery','case','workflow','mock')
assert r.get_run('recovery')['status']=='queued'
r.close()
print('restart verified')
"""
    result = subprocess.run([sys.executable, "-c", program, str(db)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "restart verified" in result.stdout


@pytest.mark.parametrize("severity,likelihood,expected", [(1, "A", "high"), (1, "E", "special_review"), (5, "E", "low"), (3, "C", "medium")])
def test_reference_risk_classes(severity, likelihood, expected):
    with tool_audit("case", lambda event: None):
        result = assess_reference_risk(severity=severity, likelihood=likelihood, evidence_ids=["E"], severity_basis="Synthetic", likelihood_basis="Synthetic")
    assert result["risk_level"] == expected and result["human_review_required"]


def test_hosted_preflight_blocks_overbudget_before_io(monkeypatch):
    from blackwing.agents.verification.agent import SPEC
    from blackwing.rag import small_context
    monkeypatch.setattr(small_context, "count_input_tokens", lambda *args, **kwargs: 9000)
    cfg = replace(settings(), llm_provider="openai_compatible", api_key="secret-sentinel")
    sent = []
    def handler(request):
        sent.append(request)
        raise AssertionError("Must not send")
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await runtime.invoke_agent(SPEC, client=client, settings=cfg, narrative="Synthetic", evidence=[], prior_outputs={})
    with pytest.raises(ValueError, match="token_budget"):
        asyncio.run(run())
    assert sent == [] and "secret-sentinel" not in repr(cfg)


def test_hosted_serialization_rate_pacing_and_transient_transport_retry(monkeypatch):
    active = 0
    peak = 0
    starts = []
    now = [100.0]
    attempts = 0
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    original_sleep = asyncio.sleep
    async def fake_sleep(delay):
        now[0] += delay
        await original_sleep(0)
    monkeypatch.setattr(runtime.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runtime, "_LAST_HOSTED_REQUEST_AT", 0)
    monkeypatch.setattr(runtime, "_HOSTED_LOCK", asyncio.Lock())
    async def handler(request):
        nonlocal active, peak, attempts
        active += 1
        peak = max(peak, active)
        starts.append(now[0])
        attempts += 1
        await original_sleep(0)
        active -= 1
        if attempts == 1:
            raise httpx.ConnectError("Synthetic transport error", request=request)
        return httpx.Response(200)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await asyncio.gather(*[runtime._hosted_post(client, url="https://iit.invalid/v1", headers={}, payload={"max_tokens": 512}) for _ in range(3)])
    responses = asyncio.run(run())
    assert all(r.status_code == 200 for r in responses)
    assert peak == 1 and attempts == 4
    assert all(b-a >= 1.05-1e-6 for a,b in zip(starts, starts[1:]))


def test_hosted_output_limit_blocks_before_request():
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: pytest.fail("No IO expected"))) as client:
            await runtime._hosted_post(client, url="https://iit.invalid", headers={}, payload={"max_tokens": 1025})
    with pytest.raises(ValueError, match="token_budget"):
        asyncio.run(run())


def test_targeted_request_reaches_model_prompt(monkeypatch):
    from blackwing.agents.registry import AGENT_BY_ID
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"recommendations": []}'}})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await runtime.invoke_agent(AGENT_BY_ID["mitigation"], client=client, settings=settings(),
                narrative="Synthetic runway conflict", evidence=[], prior_outputs={
                    "correction_request": {"target": "mitigation", "issues": [{"issue_type": "no_grounded_mitigations"}]}})
    asyncio.run(run())
    assert "no_grounded_mitigations" in captured[0]["messages"][0]["content"]
    assert "never invent evidence" in captured[0]["messages"][0]["content"]


def test_planner_unassessed_mode_skips_compliance_model(monkeypatch):
    calls = mock_agents(monkeypatch)
    from blackwing import planning
    original = planning.plan_investigation
    def decide(state):
        plan = original(state)
        plan["task_modes"]["compliance"] = "unassessed"
        return plan
    monkeypatch.setattr(service, "plan_investigation", decide)
    packets = {s.key: [DIRECT] for s in service.AGENT_SPECS}
    packets["compliance"] = [{"source_type": "regulation", "document_type": "regulation"}]
    result = asyncio.run(service.analyze_incident("Synthetic runway conflict", settings(), agent_evidence=packets))
    assert "compliance" not in calls
    assert result["agent_outputs"]["compliance"]["overall_applicability"] == "not_assessed"


@pytest.mark.parametrize("airport", ["JFK", "Austin", "Goa", "Delhi", "Synthetic"])
def test_incident_locations_share_the_bounded_workflow(monkeypatch, airport):
    mock_agents(monkeypatch)
    packets = {s.key: [DIRECT] for s in service.AGENT_SPECS}
    packets["compliance"] = []
    packets["similar_case"] = [HISTORICAL]
    result = asyncio.run(service.analyze_incident(f"{airport}: synthetic taxi clearance conflict", settings(),
                         source_incident_id="SYNTH-1", agent_evidence=packets))
    assert outcome(result)["execution_steps"] <= 10
    assert result["agent_outputs"]["risk"]["hazards"][0]["risk_index"] == "C2"
    assert result["agent_outputs"]["similar_case"]["matches"][0]["source_id"] == "OLD"


def test_no_retrieval_request_does_not_claim_retrieval_execution(monkeypatch, tmp_path):
    mock_agents(monkeypatch, impossible=True)
    index = RetrievalFixture()
    monkeypatch.setattr(api, "retriever", index)
    repo = SQLiteRepository(tmp_path / "no_retrieval.sqlite3")
    api.app.dependency_overrides[api.get_repository] = lambda: repo
    api.app.dependency_overrides[api.get_settings] = settings
    try:
        with TestClient(api.app) as client:
            case_id = client.post("/api/v1/cases", json={"narrative": "Synthetic runway crossing conflict"}).json()["case_id"]
            run_id = client.post(f"/api/v1/cases/{case_id}/runs", json={"include_retrieval": False}).json()["run_id"]
            trace = client.get(f"/api/v1/runs/{run_id}/trace").json()["events"]
            assert not any(e.get("tool") == "retrieve_evidence" for e in trace)
            assert index.calls == []
    finally:
        api.app.dependency_overrides.clear()
        repo.close()


def test_failed_agent_preserves_tool_trace_and_partial_history(monkeypatch, tmp_path):
    mock_agents(monkeypatch)
    async def fail(spec, **kwargs):
        raise httpx.ConnectError("Synthetic provider failure")
    monkeypatch.setattr(service, "invoke_agent", fail)
    monkeypatch.setattr(api, "retriever", RetrievalFixture())
    repo = SQLiteRepository(tmp_path / "failure.sqlite3")
    api.app.dependency_overrides[api.get_repository] = lambda: repo
    api.app.dependency_overrides[api.get_settings] = settings
    try:
        with TestClient(api.app) as client:
            case_id = client.post("/api/v1/cases", json={"narrative": "Synthetic runway crossing conflict"}).json()["case_id"]
            run_id = client.post(f"/api/v1/cases/{case_id}/runs", json={}).json()["run_id"]
            assert client.get(f"/api/v1/runs/{run_id}").json()["status"] == "failed"
            trace = client.get(f"/api/v1/runs/{run_id}/trace").json()["events"]
            assert any(e.get("tool") == "retrieve_evidence" for e in trace)
            assert repo.get_run(run_id)["plan"]["max_corrections"] == 1
    finally:
        api.app.dependency_overrides.clear()
        repo.close()


def test_asrs_historical_comparison_is_separate_and_explicitly_voluntary():
    from blackwing.tools import find_similar_incidents
    row = {**HISTORICAL, "document_type": "voluntary_incident_report", "source_type": "voluntary_asrs"}
    events = []
    with tool_audit("case", events.append):
        comparison = find_similar_incidents(narrative="Taxi clearance conflict", evidence=[row, DIRECT], current_incident_id="SYNTH-1")
        unsafe = find_similar_incidents(narrative="Taxi clearance conflict", evidence=[{**row, "evidence_role": "case_factual"}], current_incident_id="SYNTH-1")
    assert len(comparison["matches"]) == 1
    assert "Voluntary ASRS self-report" in " ".join(comparison["matches"][0]["differences_or_unknowns"])
    assert unsafe["matches"] == []
    assert all(event["status"] == "complete" for event in events)


def test_hosted_diagnostics_never_echo_credentials(monkeypatch, caplog):
    from blackwing.agents.registry import AGENT_BY_ID
    secret = "credential-sentinel-12345"
    cfg = replace(settings(), llm_provider="openai_compatible", api_key=secret)
    monkeypatch.setattr(runtime, "build_small_messages", lambda **kw: [{"role": "user", "content": "mock"}])
    async def post(*args, **kwargs):
        request = httpx.Request("POST", "https://iit.invalid")
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"unsupported": secret})}}]}, request=request)
    monkeypatch.setattr(runtime, "_hosted_post", post)
    monkeypatch.setattr(runtime, "verify_retry_budget", lambda *args, **kwargs: None)
    async def run():
        async with httpx.AsyncClient() as client:
            with pytest.raises(ValueError):
                await runtime.invoke_agent(AGENT_BY_ID["verification"], client=client, settings=cfg,
                                           narrative="Synthetic", evidence=[], prior_outputs={})
    asyncio.run(run())
    assert "validation_details" in caplog.text
    assert secret not in caplog.text and secret not in repr(cfg)
