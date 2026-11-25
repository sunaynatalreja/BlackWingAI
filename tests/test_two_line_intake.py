"""Compact intake metadata uses explicit labels; no model calls or date guesses."""
import asyncio

from blackwing import api
from blackwing.incident_identity import declared_intake_metadata
from blackwing.repository import MemoryRepository
from blackwing.schemas import CaseCreate

HEADER='SOURCE INCIDENT ID: DCA23LA125; INCIDENT DATE: 2023-01-13; JURISDICTION: US; AIRPORT: JFK'
PROMPT=HEADER+'\nPROMPT STRATEGY: Few-shot | Investigate this reported occurrence using cited factual evidence.'


def test_compact_and_original_headers_have_identical_metadata():
    expected={'source_incident_id':'DCA23LA125','incident_date':'2023-01-13','jurisdiction':'US','location':'JFK'}
    assert declared_intake_metadata(PROMPT)==expected
    assert declared_intake_metadata(HEADER.replace('; ','\n')+'\n\nInvestigate.')==expected
    assert declared_intake_metadata('Investigate.\n'+HEADER)=={}


def test_invalid_conflicting_and_synthetic_inputs_do_not_invent_metadata():
    result=declared_intake_metadata(HEADER+'; INCIDENT DATE: 2024-01-13\nInvestigate.')
    assert 'incident_date' not in result and result['jurisdiction']=='US'
    assert 'incident_date' not in declared_intake_metadata(HEADER.replace('2023-01-13','2023-02-31'))
    assert declared_intake_metadata('CASE ID: BW-UNSEEN-002\nORIGINAL FICTIONAL training scenario. Date unknown.')=={}


def test_case_creation_preserves_compact_metadata_without_starting_inference():
    repo=MemoryRepository()
    reply=asyncio.run(api.create_case(CaseCreate(narrative=PROMPT),repo))
    assert reply.source_incident_id=='DCA23LA125'
    assert reply.incident_date.isoformat()=='2023-01-13'
    assert reply.jurisdiction=='US' and reply.location=='JFK'
    assert not repo.runs and reply.narrative==PROMPT
