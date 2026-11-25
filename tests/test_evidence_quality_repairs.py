"""Regression checks for the saved-report failures; no hosted calls."""
import asyncio
from copy import deepcopy

import pytest

from blackwing import gold_comparison as gold, synthetic_gold as policy
from blackwing.agents.risk.evidence import ground_risk_output
from blackwing.agents.similar_case.agent import build_output
from blackwing.rag.query_text import case_query_text
from test_agentic_local import settings
from test_synthetic_report_scoring import REPORT, refs, verdict


@pytest.mark.parametrize('basis', [
    'No evidence of prior incidents or frequency of similar events.',
    'No direct evidence of loss of situational awareness.',
    'Insufficient verified data for likelihood.',
    'Likelihood remains unknown.',
    'Cannot determine severity from the supplied material.',
])
def test_self_disclaimed_rating_stays_unassessed(basis):
    output={'hazards':[{'hazard_id':'H001','description':'Potential runway conflict',
        'severity':3,'likelihood':'C','severity_basis':'Documented serious consequence exposure.',
        'likelihood_basis':basis,'evidence_ids':['FACT:1']}], 'risk_notes':'Human review required.'}
    evidence=[{'chunk_id':'FACT:1','source_type':'official_factual','excerpt':'Controller directed a go-around.'}]
    original=deepcopy(output)
    result=ground_risk_output(output,evidence)
    assert result['hazards'][0]['risk_index'] is None
    assert result['hazards'][0]['severity'] is None
    assert result['assessment_status']=='unassessed_insufficient_inputs'
    assert any('explicitly lacks support' in x for x in result['missing_inputs'])
    assert result['hazards'][0]['evidence_ids']==['FACT:1'] and output==original


def test_grounded_inputs_keep_existing_categorical_matrix():
    output={'hazards':[{'hazard_id':'H001','description':'Potential runway conflict',
        'severity':3,'likelihood':'C','severity_basis':'Documented serious consequence exposure.',
        'likelihood_basis':'Documented frequency and exposure assessment.', 'evidence_ids':['FACT:1']}]}
    result=ground_risk_output(output,[{'chunk_id':'FACT:1','source_type':'official_factual'}])
    assert result['hazards'][0]['risk_index']=='C3' and result['numeric_score'] is None


@pytest.mark.parametrize('claim,quote', [
    ('The LC controller instructed the crew to go around and fly the published missed approach.',
     'The controller failed to prioritize conflicting traffic and was distracted.'),
    ('The crew of SWA2493 called LC while N564HV was overhead.',
     'The incident involved SWA2493 and N564HV at San Diego International Airport.'),
])
def test_exact_but_irrelevant_gold_quote_cannot_score(claim,quote):
    with pytest.raises(ValueError,match='operational action'):
        gold.validate_judgment({'status':'contradicted','gold_page':1,'gold_quote':quote,
                               'rationale':'Mock judgment.'},{1:quote},claim)


def test_explicit_same_action_contradiction_remains_judgeable():
    quote='The aircraft did not cross the runway.'
    assert gold.validate_judgment({'status':'contradicted','gold_page':1,'gold_quote':quote,
        'rationale':'Explicit incompatible account.'},{1:quote},'The aircraft crossed the runway.')['status']=='contradicted'


@pytest.mark.parametrize('key',['BW-G062','BW-G069'])
def test_regulatory_quote_does_not_establish_causal_or_risk_abstention(key):
    text='No regulatory, procedural, or guidance source text was supplied or retrieved for compliance.'
    report={'generated_report':{'executive_summary':text,'sections':[]}}
    with pytest.raises(ValueError,match='policy criterion'):
        policy.validate_judgment({**verdict(),'report_quote':text},refs()[0][key],{},report)


@pytest.mark.parametrize('text', ['Install new runway signs.', 'Actions were implemented yesterday.',
                                 'We recommend synchronized position monitoring.',
                                 'No recommendations are proposed; installed new signs.'])
def test_empty_proposal_list_does_not_hide_visible_action_claims(text):
    report={'generated_report':{'executive_summary':text,'sections':[]}}
    assert policy._has_visible_actions(report)
    with pytest.raises(ValueError,match='action claims'):
        policy.validate_judgment({**verdict('not_applicable'),'report_quote':text},refs()[0]['BW-G078'],{},report)


def test_training_scenario_label_is_not_a_proposed_action():
    report={'generated_report':{'executive_summary':'ORIGINAL FICTIONAL training scenario at Alder Reach Field.'}}
    assert not policy._has_visible_actions(report)


def test_justified_deferral_has_no_points_and_no_model_call_for_action_scope(monkeypatch):
    monkeypatch.setattr(policy,'references',refs)
    calls=[]
    async def judge(config,key,*args):
        calls.append(key)
        return verdict()
    result=asyncio.run(policy.compare({},REPORT,'RUN',settings(),judge=judge))
    row=next(r for r in result['comparisons'] if r['reference_id']=='BW-G078')
    assert row['status']=='not_applicable' and row['points'] is None
    assert row['assessment_method']=='rule_based_applicability'
    assert calls==list(policy.CRITERIA)[:4]


def test_case_query_uses_scenario_instead_of_strategy_instructions():
    narrative=('CASE ID: BW-UNSEEN-002 | VERIFICATION FOCUS: Obtain evidence. | REQUEST: Use the normal workflow.\n'
        'ORIGINAL FICTIONAL training scenario at Birch Ferry Airport. A cabin call interrupted the taxi readback. '
        'No event date or country/jurisdiction is supplied. These are unverified synthetic accounts.')
    query=case_query_text(narrative)
    assert 'cabin call interrupted the taxi readback' in query
    assert 'VERIFICATION FOCUS' not in query and 'Use the normal' not in query
    seen=case_query_text('SOURCE INCIDENT ID: OPS23FA010\nPROMPT STRATEGY: Few-shot | REQUEST: '
        'Investigate San Diego OPS23FA010 on 2023-08-11. Use the normal eight-agent workflow.')
    assert seen=='Investigate San Diego OPS23FA010 on 2023-08-11.'


def test_generic_instruction_overlap_cannot_create_a_historical_analogy():
    evidence=[{'source_id':'ASRS:1','document_type':'voluntary_incident_report',
        'source_type':'voluntary_asrs','evidence_role':'historical_analogy',
        'excerpt':'This report is for review only; obtain source evidence.'}]
    narrative='A delivery van stopped near an unscaled boundary. Use the normal report schema for review.'
    assert build_output(narrative,evidence)['matches']==[]
    evidence[0]['excerpt']='A delivery van approached the boundary in another incident.'
    assert len(build_output(narrative,evidence)['matches'])==1
