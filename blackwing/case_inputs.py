"""Unverified synthetic account handling, independent of gold evaluation."""
from __future__ import annotations
import re

_SYNTHETIC = re.compile(r'BW-(?:UNSEEN|SYN)[-\w]*|ORIGINAL FICTIONAL|SIMULATED TRAINING CASE', re.I)
_CARD = re.compile(r'^\s*(BW-(?:UNSEEN|SYN)[A-Z0-9-]*-SIM-E\d+):[ \t]*(.+)$', re.I | re.M)


def is_synthetic_case(narrative):
    return bool(_SYNTHETIC.search(narrative or ''))


def synthetic_accounts(narrative, case_id):
    """Expose supplied card IDs as run-local unverified passages, never index them."""
    if not is_synthetic_case(narrative):
        return []
    cards = {}
    for match in _CARD.finditer(narrative):
        identifier, excerpt = match.group(1), match.group(2).strip()
        if identifier in cards and cards[identifier] != excerpt:
            raise ValueError('Conflicting supplied accounts reuse the same synthetic card ID.')
        cards[identifier] = excerpt
    return [{'evidence_id':identifier,'chunk_id':identifier,'source_id':identifier,
             'title':'Supplied synthetic account '+identifier,'excerpt':excerpt,
             'source_type':'user_submitted','document_type':'incident_narrative',
             'evidence_role':'reported','incident_id':case_id,
             'evidence_status':'SYNTHETIC_UNVERIFIED'} for identifier,excerpt in cards.items()]


def synthetic_risk_unassessed(state, sources):
    if not is_synthetic_case(state.get('narrative')):
        return False
    # Historical analogies and guidance cannot establish this event's ratings.
    return not any(row.get('source_type') in {'official_factual','official_preliminary'}
        and row.get('evidence_role') in {'case_factual','direct','current_case'}
        and row.get('evidence_status') != 'SYNTHETIC_UNVERIFIED'
        and str(row.get('incident_id') or '') in {
            str(state.get('source_incident_id') or state.get('case_id')),
            str(state.get('case_id'))} for row in sources)


def unassessed_synthetic_risk(state, evidence):
    from blackwing.agents.risk.schema import RiskOutput
    allowed={str(row.get('chunk_id') or row.get('evidence_id') or '') for row in evidence}
    investigation=state.get('outputs',{}).get('investigation',{})
    hazards=[]
    # Retain already stated, cited hypotheses; create no new hazard from a mere
    # proximity report and never invent rated severity or likelihood.
    for claim in [*investigation.get('hypotheses',[]),*investigation.get('contributing_factors',[])]:
        refs=[ref for ref in claim.get('evidence_ids',[]) if ref in allowed]
        if claim.get('text') and refs:
            hazards.append({'hazard_id':f'H{len(hazards)+1:03d}',
                'description':'Unverified hypothesis: '+claim['text'],'evidence_ids':refs,
                'severity':None,'likelihood':None,'severity_basis':'','likelihood_basis':''})
        if len(hazards)>=3:
            break
    missing=['Verified incident-specific severity evidence is unavailable.',
             'A defensible occurrence/exposure basis for likelihood is unavailable.']
    if not state.get('incident_date'):
        missing.append('Incident date is unknown; do not invent a date or procedure edition.')
    if not state.get('jurisdiction'):
        missing.append('Jurisdiction is unknown; binding procedural applicability requires verification.')
    return RiskOutput(hazards=hazards,risk_notes=(
        'Unassessed: only unverified synthetic accounts and/or general guidance were supplied. '
        'No categorical risk index or numerical score is justified. Human verification is required.'),
        assessment_status='unassessed_insufficient_inputs',missing_inputs=missing).model_dump(mode='json')
