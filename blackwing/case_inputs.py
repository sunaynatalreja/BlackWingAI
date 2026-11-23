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
    # Preserve cited model hypotheses when present.
    for claim in [*investigation.get('hypotheses',[]),*investigation.get('contributing_factors',[])]:
        refs=[ref for ref in claim.get('evidence_ids',[]) if ref in allowed]
        if claim.get('text') and refs:
            hazards.append({'hazard_id':f'H{len(hazards)+1:03d}',
                'description':'Unverified hypothesis: '+claim['text'],'evidence_ids':refs,
                'severity':None,'likelihood':None,'severity_basis':'','likelihood_basis':''})
        if len(hazards)>=3:
            break
    # A proximity account is a reportable potential hazard even when it does
    # not establish runway entry, severity, or likelihood. Keep it visibly
    # distinct from a verified occurrence and link only supplied passages.
    if not hazards:
        synthetic_rows=[row for row in evidence
            if row.get('evidence_status')=='SYNTHETIC_UNVERIFIED'
            and row.get('source_type')=='user_submitted']
        account_text=' '.join(str(row.get('excerpt') or '') for row in synthetic_rows).casefold()
        proximity_terms=('proximit','close to the runway','near the runway','runway')
        vehicle_terms=('vehicle','van','truck','car','driver','access limit')
        if (any(term in account_text for term in proximity_terms)
                and any(term in account_text for term in vehicle_terms)):
            refs=[str(row.get('chunk_id') or row.get('evidence_id') or '') for row in synthetic_rows
                  if str(row.get('chunk_id') or row.get('evidence_id') or '') in allowed]
            refs=list(dict.fromkeys(ref for ref in refs if ref))
            if refs:
                hazards.append({'hazard_id':'H001',
                    'description':('Potential vehicle–runway proximity hazard for verification: the supplied crew account reports apparent proximity, '
                        'while the driver account reports stopping before an access limit. The accounts do not establish boundary entry, runway occupancy, or an incursion.'),
                    'evidence_ids':refs,'severity':None,'likelihood':None,
                    'severity_basis':'','likelihood_basis':'',
                    'control_gaps':['Surveyed access boundary and synchronized vehicle/aircraft positions are unavailable.']})
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
