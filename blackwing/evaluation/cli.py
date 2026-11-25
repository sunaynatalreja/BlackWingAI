"""VS Code / PowerShell report evaluation. Defaults to a no-model preparation run."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sqlite3
from contextlib import closing

from .materials import case_materials
from .report_judge import evaluate_report, prepare_packet


def read_saved_run(database, case_id, run_id):
    """Read SQLite without invoking repository startup/recovery or changing history."""
    database = Path(database).resolve()
    if not database.is_file():
        raise ValueError('Saved SQLite database not found.')
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as connection:
        def read(kind, identifier):
            row = connection.execute('SELECT payload FROM records WHERE kind=? AND id=?', (kind, identifier)).fetchone()
            if row is None:
                raise ValueError('Saved case/run not found.')
            return json.loads(row[0])
        case, run = read('cases', case_id), read('runs', run_id)
    if run.get('case_id') != case_id or run.get('status') != 'completed' or not run.get('result'):
        raise ValueError('A completed saved run matching the case is required.')
    return case, run


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database', type=Path, default=Path('work/blackwing.sqlite3'))
    p.add_argument('--case-id')
    p.add_argument('--run-id')
    p.add_argument('--list-runs', action='store_true', help='List completed saved reports without model calls')
    p.add_argument('--manifest', type=Path, help='Optional approved held-out case/reference manifest')
    p.add_argument('--output', type=Path, help='New private evaluation directory; not a RAG source')
    p.add_argument('--execute', action='store_true', help='Opt in to hosted judge calls; consumes IIT quota')
    args = p.parse_args(argv)
    if args.list_runs:
        if not args.database.is_file():
            p.error('Saved SQLite database not found.')
        with closing(sqlite3.connect(args.database.resolve().as_uri()+'?mode=ro',uri=True)) as connection:
            saved = []
            for (payload,) in connection.execute("SELECT payload FROM records WHERE kind='runs'"):
                row = json.loads(payload)
                if row.get('status') == 'completed' and (row.get('result') or {}).get('generated_report'):
                    saved.append({'case_id':row['case_id'],'run_id':row['run_id'],
                                  'created_at':row.get('created_at'),
                                  'input_snapshot_complete':(row.get('evaluation_input_snapshot') or {}).get('complete') is True})
        print(json.dumps(saved,indent=2,default=str))
        return
    if not args.case_id or not args.run_id or not args.output:
        p.error('Supply case-id, run-id and output, or use --list-runs.')
    if args.output.exists():
        p.error('Output exists; preserve prior results.')
    case, run = read_saved_run(args.database, args.case_id, args.run_id)
    materials = case_materials(case, manifest_path=args.manifest)
    report, snapshot = run['result'], run.get('evaluation_input_snapshot') or {}
    packets = [prepare_packet(report, q, materials, snapshot) for q in materials['questions']]
    args.output.mkdir(parents=True)
    (args.output/'evaluation_packets_PRIVATE.json').write_text(json.dumps(packets, indent=2, default=str), encoding='utf-8')
    if not args.execute:
        result = {'status': 'PREPARED_NOT_EXECUTED', 'score': None, 'model_calls': 0,
                  'question_count': len(packets), 'input_snapshot_complete': snapshot.get('complete') is True,
                  'notice': 'Private evaluation-only packets. Do not upload or index this directory.'}
    else:
        from blackwing.api import get_settings
        settings = get_settings()
        def save(value):
            # CLI keeps evaluation results separate from immutable saved reports.
            temporary = args.output/'progress.tmp'
            temporary.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
            temporary.replace(args.output/'progress.json')
        result = asyncio.run(evaluate_report(report, materials, settings, run_id=args.run_id,
                                            snapshot=snapshot, save=save))
    (args.output/'result.json').write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
    print(json.dumps({k: result.get(k) for k in ('status','score','question_count','questions_completed','questions_pending')}, indent=2))


if __name__ == '__main__':
    main()
