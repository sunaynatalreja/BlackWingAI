"""Literal, attributed intake recovery; never reconstruct rejected model text."""
from blackwing.agents.evidence_utils import is_direct
from blackwing.agents.investigation.schema import InvestigationOutput
from blackwing.case_inputs import attributed_synthetic_summary, is_synthetic_case, synthetic_accounts


def supplied_account_fallback(narrative, evidence, *, case_id, incident_id=None):
    if not is_synthetic_case(narrative):
        raise ValueError('Literal recovery is restricted to explicitly synthetic scenarios.')
    if any(is_direct(row, incident_id) and row.get('source_type') in
           {'official_factual', 'official_preliminary'} for row in evidence):
        raise ValueError('Official factual analysis cannot be replaced with a synthetic intake fallback.')
    supplied={str(row.get('chunk_id') or row.get('evidence_id') or ''):row
              for row in evidence if row.get('source_type') in {'user_submitted','user_report'}
              and str(row.get('incident_id') or '')==str(case_id)}
    cards=synthetic_accounts(narrative,case_id)
    observations=[]
    if cards:
        for card in cards:
            identifier=card['evidence_id']
            if supplied.get(identifier,{}).get('excerpt')!=card['excerpt']:
                raise ValueError('Synthetic account is absent from the supplied evidence packet.')
            observations.append({'text':card['excerpt'],'evidence_state':'reported_observation',
                                 'evidence_ids':[identifier]})
    else:
        identifier='CASE-NARRATIVE:'+str(case_id)
        if supplied.get(identifier,{}).get('excerpt')!=narrative.strip():
            raise ValueError('The original submitted narrative is absent from the evidence packet.')
        observations=[{'text':attributed_synthetic_summary(narrative),
                       'evidence_state':'reported_observation','evidence_ids':[identifier]}]
    return InvestigationOutput(
        incident_summary=attributed_synthetic_summary(narrative),
        reported_observations=observations,
        unknowns=[
            'Investigation model analysis was unavailable after output validation failed. '
            'Literal supplied accounts were preserved as unverified observations; human review is required.',
            'Authenticated source records are required to establish the event sequence, exact communications, positions and cause.',
        ],
    ).model_dump(mode='json')
