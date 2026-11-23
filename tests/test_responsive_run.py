"""Retrieval must not block run acceptance or the API event loop."""
import asyncio
import threading
import time

import pytest
from fastapi import BackgroundTasks
from streamlit.testing.v1 import AppTest

from blackwing import api, copilot
from blackwing.repository import MemoryRepository
from blackwing.schemas import RunCreate
from test_agentic_local import settings
from test_streamlit_integration import backend, synthetic_runner, ROOT


def test_run_acceptance_does_not_execute_retrieval(monkeypatch):
    repo=MemoryRepository()
    case=repo.create_case('case',{'narrative':'Synthetic runway crossing for response test.','category':'runway_incursion'})
    monkeypatch.setattr(api.retriever,'search',lambda *a,**k:pytest.fail('Retrieval must start after acceptance'))
    background=BackgroundTasks()
    reply=asyncio.run(api.create_analysis_run(case_id='case',request=RunCreate(),background=background,
        repo=repo,settings=settings(),runner=synthetic_runner))
    assert reply.status=='queued' and len(background.tasks)==1
    assert repo.get_case('case')['status']=='analysis_running'
    assert not repo.get_run(reply.run_id).get('audit_events')


@pytest.mark.parametrize('phase',['initial','agent_packets'])
def test_waiting_for_retrieval_keeps_health_and_persisted_progress_responsive(monkeypatch,phase):
    repo=MemoryRepository()
    case=repo.create_case('case',{'narrative':'Synthetic runway crossing response test.','category':'runway_incursion'})
    repo.create_run('run','case','runway_incursion_investigation','mock')
    entered=threading.Event()
    release=threading.Event()
    def search(*args,**kwargs):
        should_wait=kwargs.get('scope')=='direct' if phase=='agent_packets' else 'scope' not in kwargs
        if should_wait:
            entered.set()
            assert release.wait(3),'Test did not release retrieval worker'
        return {'results':[]}
    monkeypatch.setattr(api.retriever,'search',search)
    monkeypatch.setenv('BLACKWING_AGENT_RAG','1' if phase=='agent_packets' else '0')
    async def run():
        task=asyncio.create_task(api._execute_run('run',case,None if phase=='initial' else [],
            settings(),repo,synthetic_runner,True))
        try:
            start=time.monotonic()
            while not entered.is_set() and time.monotonic()-start<1:
                await asyncio.sleep(0.01)
            assert entered.is_set()
            assert time.monotonic()-start<1
            assert (await asyncio.wait_for(api.health(),timeout=0.2))['status']=='ok'
            assert repo.get_run('run')['status']=='running'
            assert any(e['status']=='retrieval_started' for e in repo.get_run('run')['audit_events'])
        finally:
            release.set()
            await task
        assert repo.get_run('run')['status']=='completed'
    asyncio.run(run())


def test_running_case_has_visible_progress_and_does_not_resubmit(backend):
    case=copilot.create_case(narrative='Synthetic runway crossing progress display test.')
    backend.create_run('waiting-run',case['case_id'],'runway_incursion_investigation','mock')
    backend.append_event('waiting-run',{'event_id':'progress','step':'evidence_retrieval',
        'status':'retrieval_started'})
    at=AppTest.from_file(str(ROOT/'blackwing/ui.py'),default_timeout=15)
    at.query_params['case_id']=case['case_id']
    at.run()
    assert not at.exception
    assert at.button(key='analyze-incident').disabled
    assert not any('Report unavailable: the investigation is still running' in item.value for item in at.info)
    assert not any(b.label == 'Refresh investigation status' for b in at.button)
    metrics={m.label:m.value for m in at.metric}
    assert metrics['Gold comparison score']=='Pending' and metrics['Aviation risk']=='Pending'
    assert any('Retrieving evidence' in item.value for item in at.text)
    at.run()  # A polling rerun must read status without submitting another run.
    assert at.button(key='analyze-incident').disabled
    assert len(backend.cases)==1 and len(backend.runs)==1
