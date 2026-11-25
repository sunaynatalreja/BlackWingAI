"""Offline, scope-explicit Recall@5/10 from approved qrels and ranked IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def score_rankings(qrels, rankings):
    if qrels.get('unit') not in {'chunk', 'document'}:
        raise ValueError('Declare chunk or document ranking units.')
    if qrels.get('approval', {}).get('status') != 'APPROVED':
        raise ValueError('Relevance judgments require approval.')
    for field in ('index_signature', 'scope_id', 'unit'):
        if not qrels.get(field) or qrels[field] != rankings.get(field):
            raise ValueError('Ranking/qrels identity mismatch: ' + field)
    eligible = set(qrels.get('eligible_ids') or [])
    if not eligible or len(eligible) != len(qrels['eligible_ids']):
        raise ValueError('Eligible corpus IDs must be nonempty and unique.')
    questions = {q['question_id']: q for q in qrels['questions']}
    if len(questions) != len(qrels['questions']):
        raise ValueError('Duplicate question IDs.')
    methods = sorted({r['method'] for r in rankings['queries']})
    if not methods:
        raise ValueError('No ranked methods supplied.')
    lookup = {}
    for r in rankings['queries']:
        key = (r['question_id'], r['method'])
        if key in lookup or key[0] not in questions:
            raise ValueError('Duplicate or unrecognized ranked query.')
        lookup[key] = r
    details = []
    for qid, q in questions.items():
        relevant = set(q.get('relevant_ids') or [])
        if not relevant <= eligible:
            raise ValueError('Unknown or out-of-scope relevant ID: ' + qid)
        for method in methods:
            row = lookup.get((qid, method))
            if row is None:
                raise ValueError('Missing ranked query; an explicit empty result is required: ' + qid)
            ranked = row.get('ranked_ids') or []
            if len(ranked) != len(set(ranked)):
                raise ValueError('Duplicate ranked IDs; do not silently refill top k.')
            if not set(ranked) <= eligible:
                raise ValueError('Unknown or out-of-scope retrieved ID.')
            actual = row.get('actual_method') or method
            ok = row.get('status', 'completed') == 'completed' and actual == method
            complete = q.get('complete_relevance') is True
            status = ('provider_or_method_failure' if not ok else 'incomplete_judgments' if not complete
                      else 'no_eligible_relevant_items' if not relevant else 'measured')
            result = {'question_id': qid, 'method': method, 'actual_method': actual,
                      'status': status, 'relevant_count': len(relevant)}
            for k in (5, 10):
                hits = len(set(ranked[:k]) & relevant)
                result[f'relevant_retrieved@{k}'] = hits
                result[f'recall@{k}'] = hits / len(relevant) if status == 'measured' else None
                result[f'hit@{k}'] = int(hits > 0) if status == 'measured' else None
            details.append(result)
    summaries = []
    for method in methods:
        rows = [r for r in details if r['method'] == method]
        measured = [r for r in rows if r['status'] == 'measured']
        summaries.append({'method': method, 'queries_total': len(rows), 'queries_measured': len(measured),
            'judgment_coverage': len(measured) / len(rows) if rows else None,
            'recall@5': mean(r['recall@5'] for r in measured) if measured else None,
            'recall@10': mean(r['recall@10'] for r in measured) if measured else None})
    return {'metric': 'Recall@k', 'unit': qrels['unit'], 'index_signature': qrels['index_signature'],
            'scope_id': qrels['scope_id'], 'scope_description': qrels.get('scope_description'),
            'formula': 'relevant IDs in the first k ranked results / all approved relevant IDs in the declared scope',
            'summary': summaries, 'queries': details,
            'notice': 'Retrieval evaluation only. Not report accuracy, structural completeness or aviation risk.'}


def bm25_replay(index_dir, qrels):
    """Read existing chunks and rank a declared subset; never build/save an index."""
    from blackwing.rag.engine import lexical_rank, _is_protected_gold
    index_dir = Path(index_dir)
    manifest = json.loads((index_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['signature'] != qrels['index_signature']:
        raise ValueError('Qrels belong to a different index signature.')
    eligible = set(qrels['eligible_ids'])
    unit = qrels['unit']
    rows, represented = [], set()
    with (index_dir / 'chunks.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            identifier = row['chunk_id'] if unit == 'chunk' else row.get('source_id')
            if identifier not in eligible:
                continue
            if _is_protected_gold(row):
                raise ValueError('A protected source was included in the evaluation corpus.')
            rows.append(row)
            represented.add(identifier)
    if represented != eligible:
        raise ValueError('Qrels contain IDs absent from the existing index.')
    output = {'index_signature': manifest['signature'], 'scope_id': qrels['scope_id'],
              'unit': unit, 'ranking_rule': 'BM25; documents ordered by their best-scoring chunk', 'queries': []}
    for q in qrels['questions']:
        ordered, seen = [], set()
        for _, i in lexical_rank(q['query'], rows):
            identifier = rows[i]['chunk_id'] if unit == 'chunk' else rows[i]['source_id']
            if identifier not in seen:
                ordered.append(identifier)
                seen.add(identifier)
        output['queries'].append({'question_id': q['question_id'], 'method': 'bm25',
            'actual_method': 'bm25', 'status': 'completed', 'ranked_ids': ordered})
    return output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qrels', type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--bm25-index', type=Path, help='Existing index; read only, no embeddings')
    group.add_argument('--rankings', type=Path, help='Compatible saved rankings; no provider calls')
    p.add_argument('--output', type=Path, required=True, help='New directory; refuses overwrite')
    args = p.parse_args(argv)
    if args.output.exists():
        p.error('Output exists; preserve the prior evaluation.')
    qrels = json.loads(args.qrels.read_text(encoding='utf-8'))
    rankings = (bm25_replay(args.bm25_index, qrels) if args.bm25_index else
                json.loads(args.rankings.read_text(encoding='utf-8')))
    result = score_rankings(qrels, rankings)
    args.output.mkdir(parents=True)
    for name, data in [('rankings.json', rankings), ('metrics.json', result)]:
        (args.output / name).write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(json.dumps({'unit': result['unit'], 'scope_id': result['scope_id'], 'summary': result['summary']}, indent=2))


if __name__ == '__main__':
    main()
