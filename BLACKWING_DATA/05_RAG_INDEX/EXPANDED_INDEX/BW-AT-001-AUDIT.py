#!/usr/bin/env python3
"""Offline SQLite RAG audit. No model inference, network or application modifications."""
from pathlib import Path
import csv, sqlite3, re, json
ROOT=Path(__file__).resolve().parents[2]
DB=ROOT/'05_RAG_INDEX/EXPANDED_INDEX/BW-IX-001-FTS5-TRIAL.sqlite'
ROUTES=ROOT/'01_RAG_KNOWLEDGE_BASE/BW-AR-001-SEVEN-AGENT-ROUTING.csv'
con=sqlite3.connect(f'file:{DB.as_posix()}?mode=ro',uri=True)
try:
    checks={
      'sqlite_integrity':con.execute('PRAGMA integrity_check').fetchone()[0],
      'sources':con.execute('SELECT COUNT(*) FROM sources').fetchone()[0],
      'chunks':con.execute('SELECT COUNT(*) FROM chunks').fetchone()[0],
      'indexed_fts_rows':con.execute('SELECT COUNT(*) FROM fts').fetchone()[0],
      'agent_routes':dict(con.execute('SELECT agent,COUNT(DISTINCT source_id) FROM routes GROUP BY agent').fetchall()),
      'protected_folder_sources':con.execute("SELECT COUNT(*) FROM sources WHERE path LIKE '02_GOLD_EVALUATION_PROTECTED/%'").fetchone()[0],
      'missing_citation_ids':con.execute("SELECT COUNT(*) FROM chunks WHERE chunk_id IS NULL OR chunk_id = '' OR page IS NULL").fetchone()[0],
      'corpus_status':con.execute("SELECT value FROM index_meta WHERE key='corpus_status'").fetchone()[0]
    }
    expected={'investigation','similar_case','compliance','risk','mitigation','critic','verification'}
    assert checks['sqlite_integrity']=='ok'
    assert checks['sources']==49 and checks['chunks']>=20000
    assert checks['chunks']==checks['indexed_fts_rows']
    assert expected==set(checks['agent_routes'])
    assert checks['protected_folder_sources']==0
    assert checks['missing_citation_ids']==0
    assert ROUTES.is_file()
    print(json.dumps({'result':'PASS','checks':checks},indent=2))
finally:
    con.close()
