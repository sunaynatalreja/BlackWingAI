"""Durable diagnostics and recovery; provider IO is exclusively MockTransport."""
import asyncio
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from blackwing import api, copilot, service
from blackwing.diagnostics import lifecycle_event
from blackwing.repository import MemoryRepository, SQLiteRepository
from test_agentic_local import settings, mock_agents
from test_streamlit_integration import backend

ROOT = Path(__file__).resolve().parents[1]


def raw_run(db, run_id):
    with closing(sqlite3.connect(db)) as connection:
        cursor = connection.execute("SELECT payload FROM records WHERE kind='runs' AND id=?", (run_id,))
        try:
            payload = cursor.fetchone()[0]
        finally:
            cursor.close()
        return json.loads(payload)


def case_record(repo):
    return repo.create_case("case", {"narrative": "Synthetic runway crossing with uncertain clearance.", "category": "runway_incursion"})


def install_mock_transport(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(service.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


def test_started_and_waiting_commit_before_model_returns(monkeypatch, tmp_path):
    db = tmp_path / "incremental.sqlite3"
    repo = SQLiteRepository(db)
    case_record(repo)
    repo.create_run("run", "case", "runway_incursion_investigation", "mock")
    repo.patch_run("run", {"status": "running"})
    mock_agents(monkeypatch)
    fake = service.invoke_agent
    from blackwing.agents.runtime import invoke_agent as real_invoke
    async def select(spec, **kwargs):
        return await real_invoke(spec, **kwargs) if spec.key == "investigation" else await fake(spec, **kwargs)
    monkeypatch.setattr(service, "invoke_agent", select)
    observed = []
    def handler(request):
        saved = raw_run(db, "run")
        events = saved["audit_events"]
        assert [e["status"] for e in events if e.get("agent_key") == "investigation"] == ["agent_started", "waiting_for_model_response"]
        assert saved["progress"][-1]["status"] == "waiting_for_model_response"
        assert not any(e["status"] == "agent_completed" for e in events)
        observed.append(True)
        return httpx.Response(200, json={"message": {"content": '{"incident_summary":"Synthetic event"}'}})
    install_mock_transport(monkeypatch, handler)
    result = asyncio.run(service.analyze_incident("Synthetic runway crossing with uncertain clearance.", settings(), case_id="case",
                         on_execution_event=lambda event: repo.append_event("run", event)))
    assert observed == [True]
    events = repo.get_run("run")["audit_events"]
    started = [e for e in events if e["status"] == "agent_started"]
    completed = [e for e in events if e["status"] == "agent_completed"]
    assert {e["agent_key"] for e in completed} == {spec.key for spec in service.AGENT_SPECS}
    assert len(started) == len(completed) == 8
    assert {e["execution_id"] for e in started} == {e["execution_id"] for e in completed}
    assert all(e["duration_ms"] >= 0 for e in completed)
    # Deterministic roles must not claim an outbound model request.
    assert not any(e["status"] == "waiting_for_model_response" and e["agent_key"] in {"similar_case", "compliance", "report"} for e in events)
    repo.patch_run("run", {"status": "completed", "result": result})
    repo.close()
    restarted = SQLiteRepository(db)
    assert len([e for e in restarted.get_run("run")["audit_events"] if e["status"] == "agent_completed"]) == 8
    assert restarted.get_run("run")["status"] == "completed"
    restarted.close()


@pytest.mark.parametrize("kind,expected", [("timeout", "model_timeout"), ("http", "model_http_error"), ("unexpected", "agent_processing_error")])
def test_provider_failures_persist_and_api_remains_available(monkeypatch, backend, caplog, kind, expected):
    secret = "NEVER-LOG-THIS-CREDENTIAL"
    api.app.dependency_overrides[api.get_analysis_runner] = lambda: service.analyze_incident
    monkeypatch.setenv("BLACKWING_AGENT_RAG", "0")
    def handler(request):
        if kind == "timeout":
            raise httpx.ReadTimeout(secret, request=request)
        if kind == "http":
            return httpx.Response(503, json={"error": secret}, headers={"secret": secret})
        raise RuntimeError(secret)
    install_mock_transport(monkeypatch, handler)
    with TestClient(api.app) as client:
        case = client.post("/api/v1/cases", json={"narrative": "Synthetic runway crossing with uncertain clearance."}).json()
        accepted = client.post(f"/api/v1/cases/{case['case_id']}/runs", json={"include_retrieval": False})
        assert accepted.status_code == 202
        run_id = accepted.json()["run_id"]
        assert client.get("/health").status_code == 200
        run = client.get(f"/api/v1/runs/{run_id}").json()
        assert run["status"] == "failed" and run["error_code"] == expected
        events = client.get(f"/api/v1/runs/{run_id}/trace").json()["events"]
        assert any(e["status"] == "agent_started" for e in events)
        assert any(e["status"] == "waiting_for_model_response" for e in events)
        failed = next(e for e in events if e["status"] == "agent_failed")
        assert failed["duration_ms"] >= 0 and failed["error"]["code"] == expected
        assert any(e["status"] == "execution_failed" for e in events)
        assert not any(e["status"] == "agent_completed" for e in events)
        assert secret not in json.dumps(events) and secret not in caplog.text
        if kind == "http":
            assert failed["error"]["http_status"] == 503
        assert client.post("/api/v1/cases", json={"narrative": "A second independent synthetic incident after failure."}).status_code == 201


def test_system_exit_attempt_is_contained_inside_graph_task(monkeypatch, backend):
    api.app.dependency_overrides[api.get_analysis_runner] = lambda: service.analyze_incident
    async def exit_attempt(spec, **kwargs):
        raise SystemExit("secret exit details")
    monkeypatch.setattr(service, "invoke_agent", exit_attempt)
    case = copilot.create_case(narrative="Synthetic event with a model adapter exit attempt.")
    accepted = copilot.start_investigation(case["case_id"], include_retrieval=False)
    run = copilot.get_run(accepted["run_id"])
    assert run["status"] == "failed" and run["error_code"] == "process_exit_attempt"
    assert copilot.backend_status()["status"] == "ok"
    assert "secret exit details" not in json.dumps(copilot.get_trace(run["run_id"]))


def test_setup_and_failure_bookkeeping_exceptions_do_not_escape(caplog):
    class BrokenRepository(MemoryRepository):
        def patch_run(self, *args, **kwargs):
            raise OSError("secret storage path")
    repo = BrokenRepository()
    case = case_record(repo)
    repo.create_run("run", "case", "runway_incursion_investigation", "mock")
    async def must_not_call(*args, **kwargs):
        pytest.fail("Runner must not start after setup failure")
    asyncio.run(api._execute_run("run", case, [], settings(), repo, must_not_call, False))
    assert "Could not persist background failure" in caplog.text
    assert "secret storage path" not in caplog.text
    assert repo.get_run("run")["audit_events"][-1]["status"] == "execution_failed"


def test_cancellation_records_interruption_and_propagates(monkeypatch, tmp_path):
    repo = SQLiteRepository(tmp_path / "cancel.sqlite3")
    case = case_record(repo)
    repo.create_run("run", "case", "runway_incursion_investigation", "mock")
    async def cancel(spec, **kwargs):
        raise asyncio.CancelledError()
    monkeypatch.setattr(service, "invoke_agent", cancel)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(api._execute_run("run", case, [], settings(), repo, service.analyze_incident, False))
    saved = repo.get_run("run")
    assert saved["error_code"] == "interrupted" and saved["status"] == "failed"
    assert any(e["status"] == "execution_interrupted" and e.get("agent_key") == "investigation" for e in saved["audit_events"])
    repo.close()


def test_abrupt_process_stop_preserves_waiting_and_restart_marks_interrupted(tmp_path):
    db = tmp_path / "crash.sqlite3"
    marker = tmp_path / "request-entered"
    child = tmp_path / "blocked_model.py"
    child.write_text('''
import asyncio,json,sys
from pathlib import Path
import httpx
from blackwing import service,api
from blackwing.repository import SQLiteRepository
from test_agentic_local import settings
repo=SQLiteRepository(sys.argv[1])
case=repo.create_case('case',{'narrative':'Synthetic runway crossing with uncertain clearance.','category':'runway_incursion'})
repo.create_run('run','case','runway_incursion_investigation','mock')
original=httpx.AsyncClient
async def handler(request):
    Path(sys.argv[2]).write_text('waiting',encoding='utf-8')
    await asyncio.Event().wait()
service.httpx.AsyncClient=lambda **kw:original(transport=httpx.MockTransport(handler),**kw)
asyncio.run(api._execute_run('run',case,[],settings(),repo,service.analyze_incident,False))
''', encoding="utf-8")
    import os
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "tests")]), "BLACKWING_AGENT_RAG": "0"}
    with (tmp_path / "child.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(child), str(db), str(marker)], cwd=ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + 15
            while not marker.exists() and time.monotonic() < deadline and process.poll() is None:
                time.sleep(0.05)
            assert marker.exists(), (tmp_path / "child.log").read_text(encoding="utf-8")
            before = raw_run(db, "run")
            assert before["status"] == "running"
            waiting = next(e for e in before["audit_events"] if e["status"] == "waiting_for_model_response")
        finally:
            if process.poll() is None:
                process.terminate()  # real abrupt process stop; no Python cleanup
            process.wait(timeout=10)
    # Allow forced-process cleanup before reopening SQLite mapped files on Windows.
    time.sleep(0.1)
    try:
        restarted = SQLiteRepository(db)
    except sqlite3.OperationalError as exc:
        pytest.fail(f"Restart SQLite error {exc.sqlite_errorcode}/{exc.sqlite_errorname}: {exc}")
    saved = restarted.get_run("run")
    assert saved["status"] == "failed" and saved["error_code"] == "interrupted"
    interrupted = saved["audit_events"][-1]
    assert interrupted["status"] == "execution_interrupted"
    assert interrupted["execution_id"] == waiting["execution_id"]
    assert interrupted["agent_key"] == "investigation"
    assert not any(e["status"] in {"agent_completed", "agent_failed"} for e in saved["audit_events"])
    count = len(saved["audit_events"])
    restarted.close()
    second = SQLiteRepository(db)
    assert len(second.get_run("run")["audit_events"]) == count  # restart is idempotent
    second.create_run("recovery", "case", "runway_incursion_investigation", "mock")
    assert second.get_run("recovery")["status"] == "queued"
    second.close()


@pytest.mark.parametrize("status,error_code,word", [("queued", None, "running"), ("running", None, "running"),
    ("failed", "model_timeout", "failed"), ("failed", "interrupted", "interrupted")])
def test_report_409_explains_state_in_api_and_streamlit(backend, status, error_code, word):
    case = copilot.create_case(narrative="Synthetic incident with an unavailable report.")
    backend.create_run("run", case["case_id"], "runway_incursion_investigation", "mock")
    backend.patch_run("run", {"status": status, "error_code": error_code})
    if status == "running":
        backend.append_event("run", lifecycle_event("agent_execution", "waiting_for_model_response", case_id=case["case_id"], agent_key="investigation"))
    with TestClient(api.app) as client:
        response = client.get(f"/api/v1/cases/{case['case_id']}/report")
        assert response.status_code == 409 and word in response.json()["detail"]
    projected = copilot.get_case(case["case_id"])
    assert projected["report"] is None and word in projected["report_availability"]["message"]
    at = AppTest.from_file(str(ROOT / "blackwing/ui.py"), default_timeout=15)
    at.session_state["workspace_view"] = "case"
    at.session_state["workspace_case"] = case["case_id"]
    at.run()
    assert not at.exception
    assert any("Report unavailable" in a.value and word in a.value for a in [*at.info, *at.warning])
    if status == "running":
        assert any("Waiting for model response" in e.value for e in at.text)


def test_lazy_repository_initialization_failure_is_also_contained(caplog):
    class UnavailableRepository:
        def __getattr__(self, key):
            raise OSError("secret database location")
    async def must_not_run(*args, **kwargs):
        pytest.fail("No runner call expected")
    asyncio.run(api._execute_run("run", {"case_id": "case"}, [], settings(), UnavailableRepository(), must_not_run, False))
    assert "Could not persist background failure" in caplog.text
    assert "secret database location" not in caplog.text
