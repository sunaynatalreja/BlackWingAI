#!/usr/bin/env python3
"""BLACKWING seven-agent, source-scoped sparse evidence retrieval and smoke testing.
No model calls. Uses SQLite FTS5 built from user-supplied source PDFs. Treat as a trial index, not approved findings.
Example: python BW-RT-001-RETRIEVE.py --smoke
         python BW-RT-001-RETRIEVE.py --agent investigation --query "taxi route runway 4L" --case JFK
"""
import argparse,csv,json,re,sqlite3,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # BLACKWING_DATA
DB=ROOT/'05_RAG_INDEX/EXPANDED_INDEX/BW-IX-001-FTS5-TRIAL.sqlite'
SMOKE=ROOT/'06_EVALUATION_RESULTS/RETRIEVAL_METRICS/BW-RQ-001-SEVEN-AGENT-SMOKE-QUERIES.csv'
ROLES={'investigation','similar_case','compliance','risk','mitigation','critic','verification'}
ALIASES={'JFK':'DCA23LA125','AUSTIN':'DCA23FA149','GOA':'IND-AAIB-001-VT-BXL','DELHI2013':'IND-AAIB-002-VT-GOK','DELHI2025':'IND-AAIB-003-AFG311'}
STOP={'that','what','when','which','with','were','from','was','for','the','and','this','during','their','about','into','how','does','did','are','have','has','its','his','her','they','could','would','been','flight','incident','what','where','who'}

def query_fts(q):
    terms=[]
    for t in re.findall(r'[A-Za-z0-9]+',q.lower()):
        if len(t)<3 or t in STOP or t in terms:continue
        terms.append(t)
    if not terms:return None
    return ' OR '.join('"'+t+'"' for t in terms[:18])

def retrieve(agent,query,case=None,current_incident_id=None,jurisdiction=None,incident_date=None,limit=5):
    if agent not in ROLES:raise ValueError('Unknown agent '+str(agent))
    if limit<1 or limit>50:raise ValueError('Limit 1..50')
    match=query_fts(query)
    if not match:return []
    inc=(current_incident_id or ALIASES.get((case or '').upper()) or '').upper()
    conn=sqlite3.connect(f'file:{DB.as_posix()}?mode=ro',uri=True)
    conn.row_factory=sqlite3.Row
    # Enforce role-specific routes, no-final protection via materialized docs, current-case scope,
    # and jurisdiction/date relevance. Similar-case returns other-case factual packets.
    cond=['fts MATCH ?','r.agent = ?']
    args=[match,agent]
    if inc and agent=='investigation':
        cond.append('(s.incident_id = ? OR s.incident_id IS NULL)');args.append(inc)
    if inc and agent=='similar_case':
        cond.append('(s.incident_id IS NOT NULL AND s.incident_id <> ?)');args.append(inc)
    if jurisdiction and agent=='compliance':
        cond.append('(s.jurisdiction = ? OR s.jurisdiction = ? OR s.jurisdiction IS NULL)');args.extend([jurisdiction,'INTL'])
        # Compliance cannot cite a regulation/order published after occurrence as though it applied then.
        if incident_date:
            cond.append("(s.source_type NOT IN ('operational_order','regulation') OR s.publication_date IS NULL OR s.publication_date <= ?)");args.append(incident_date)
    if agent=='compliance':
        cond.append("s.source_type IN ('operational_order','regulation')")
    if agent=='similar_case':
        cond.append("s.source_type IN ('official_factual','official_preliminary')")
    if agent=='investigation':
        cond.append("s.source_type IN ('official_factual','official_preliminary')")
    if agent=='risk':
        cond.append("s.source_type IN ('official_factual','official_preliminary','advisory')")
    if agent=='mitigation':
        cond.append("s.source_type IN ('official_factual','official_preliminary','advisory')")
    sql=('SELECT c.chunk_id,c.source_id,c.page,c.part,c.text,s.path,s.source_type,s.jurisdiction,s.incident_id,s.publication_date,s.index_policy,bm25(fts) AS rank '
         'FROM fts JOIN chunks c ON c.id=fts.rowid JOIN sources s ON s.source_id=c.source_id JOIN routes r ON r.source_id=s.source_id WHERE '
         + ' AND '.join(cond) + ' ORDER BY rank LIMIT ?')
    args.append(limit)
    try:
        rows=conn.execute(sql,args).fetchall()
        return [dict(x) for x in rows]
    finally:conn.close()

def smoke():
    ok=0
    with SMOKE.open(encoding='utf-8-sig',newline='') as f: cases=list(csv.DictReader(f))
    print('BLACKWING 7-AGENT RETRIEVAL SMOKE (SPARSE TRIAL / NO LLM)')
    for entry in cases:
        rows=retrieve(entry['agent'],entry['query'],current_incident_id=entry['current_incident_id'],jurisdiction=entry['jurisdiction'],incident_date=entry['incident_date'],limit=5)
        status='PASS' if rows else 'EMPTY'
        # PASS = nonempty eligible results only; not gold correctness or human source approval.
        if rows:ok+=1
        print(f"{entry['agent']:14} {status:5} {len(rows)} top-source={(rows[0]['source_id']+':p'+str(rows[0]['page'])) if rows else '-'}")
    print(f'RETRIEVAL ROUTING: {ok}/{len(cases)} nonempty; factual correctness NOT scored')
    return ok==len(cases)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--agent',choices=sorted(ROLES))
    ap.add_argument('--query')
    ap.add_argument('--case')
    ap.add_argument('--incident-id')
    ap.add_argument('--jurisdiction')
    ap.add_argument('--incident-date')
    ap.add_argument('--limit',type=int,default=5)
    a=ap.parse_args()
    if a.smoke:sys.exit(0 if smoke() else 2)
    if not a.agent or not a.query:ap.error('Specify --smoke or --agent and --query')
    results=retrieve(a.agent,a.query,a.case,a.incident_id,a.jurisdiction,a.incident_date,a.limit)
    print(json.dumps(results,indent=2,ensure_ascii=False))
