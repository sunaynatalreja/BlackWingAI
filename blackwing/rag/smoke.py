from __future__ import annotations

import argparse
import json
import os
from .agent_router import AGENT_ROLES, build_agent_packets
from .retriever import RAGRetriever

# FICTIONAL examples deliberately avoid turning a particular historical case
# into a hard-coded retrieval dependency.
DOCS = [
 {'document_id':'DEMO-CASE-A-P1','source_id':'DEMO-CASE-A','incident_id':'DEMO-A',
  'incident_date':'2024-01-05','source_type':'official_factual','evidence_role':'case_factual',
  'document_type':'incident_report','jurisdiction':'US','page_number':1,
  'title':'Synthetic incident A',
  'content':'Synthetic exercise: an aircraft reported crossing a runway holding position; the tower clearance transcript is missing.'},
 {'document_id':'DEMO-CASE-B-P1','source_id':'DEMO-CASE-B','incident_id':'DEMO-B',
  'incident_date':'2023-02-02','source_type':'official_final','evidence_role':'historical_analogy',
  'document_type':'incident_report','jurisdiction':'US','page_number':1,
  'title':'Synthetic historical incident B',
  'content':'Synthetic exercise: a second flight entered an occupied runway and initiated a rejected takeoff. Separate historical event.'},
 {'document_id':'DEMO-US-PROC','source_id':'DEMO-US-PROC', 'source_type':'operational_order',
  'document_type':'procedure','jurisdiction':'US','effective_date':'2022-01-01',
  'title':'Synthetic dated US rule', 'content':'Synthetic procedure: runway crossing authorization requires an explicit ATC clearance.'},
 {'document_id':'DEMO-US-FUTURE','source_id':'DEMO-US-FUTURE', 'source_type':'operational_order',
  'document_type':'procedure','jurisdiction':'US','effective_date':'2026-01-01',
  'title':'Synthetic future rule', 'content':'Synthetic future procedure: runway crossing new alerting protocol.'},
 {'document_id':'DEMO-SAFETY','source_id':'DEMO-SAFETY', 'source_type':'advisory',
  'document_type':'guidance','jurisdiction':'international','publication_date':'2022-01-01',
  'title':'Synthetic aviation advisory', 'content':'Synthetic advisory: surface surveillance, runway signage and readback awareness can be safety barriers.'}
]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real-models',action='store_true')
    a=p.parse_args(argv)
    # This is an explicit local demonstration; no remote IIT endpoint is read.
    os.environ['BLACKWING_RAG_BACKEND']='local' if a.real_models else 'none'
    os.environ['BLACKWING_RAG_RERANKER']='crossencoder' if a.real_models else 'none'
    rag=RAGRetriever()
    packets=build_agent_packets(retriever=rag, narrative='Synthetic exercise: aircraft crossed runway hold line while tower frequency was busy.',
          case_id='DEMO-A', incident_date='2024-01-05', jurisdiction='US',
          documents=DOCS, embedding_model='BAAI/bge-small-en-v1.5',
          limit_per_agent=3,method='rerank' if a.real_models else 'bm25')
    check = {role:[{'source':x.get('source_id'),'citation':x.get('citation_label'),
                    'role':x.get('evidence_role')} for x in rows]
                  for role, rows in packets.items()}
    assert set(packets)==set(AGENT_ROLES)
    assert all(x['source']!='DEMO-US-FUTURE' for data in check.values() for x in data)
    assert all(x['source']!='DEMO-CASE-B' for x in check['investigation'])
    assert all(x['source']!='DEMO-CASE-A' for x in check['similar_case'])
    assert check['compliance'], 'Expected old valid synthetic rule in compliance results'
    if a.real_models:
        # Fail rather than mislabel fallback lexical data as verified BGE+CrossEncoder.
        result=rag.search('runway clearance',DOCS,method='rerank')
        if not result['results'] or result['metrics']['reranker_status']!='crossencoder' or result['metrics']['dense_candidate_count'] == 0:
            raise RuntimeError('Real BGE/CrossEncoder did not activate; inspect model downloads')
    print(json.dumps({'mode':'real_local_models' if a.real_models else 'offline_bm25',
        'seven_agent_packets':check,'structural_validation':'PASS',
        'retrieval_accuracy':'NOT VERIFIED on real corpus'},indent=2))

if __name__=='__main__': main()
