"""Offline, scope-explicit Recall@5/10 from approved qrels and ranked IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def query_with_case_context(question):
    """Keep the question generic; attach structured facts supplied by its case intake."""
    query = str(question.get('query') or '').strip()
    context = question.get('case_context') or {}
    fields = ('incident_id', 'airport', 'incident_date', 'jurisdiction',
              'flight_numbers', 'aircraft_types', 'runway')
    values = []
    for field in fields:
        value = context.get(field)
        if isinstance(value, (list, tuple)):
            values.extend(str(item).strip() for item in value if str(item).strip())
        elif value is not None and str(value).strip():
            values.append(str(value).strip())
    return ' '.join([query, *values]).strip()


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
    universe_ids = rankings.get('retrieval_universe_ids') or sorted(eligible)
    universe = set(universe_ids)
    if not universe or len(universe) != len(universe_ids) or not eligible <= universe:
        raise ValueError('The retrieval universe must uniquely contain the judged source pool.')
    full_corpus_recall_available = (
        qrels.get('full_corpus_relevance_complete') is True and eligible == universe
    )
    questions = {q['question_id']: q for q in qrels['questions']}
    if len(questions) != len(qrels['questions']):
        raise ValueError('Duplicate question IDs.')
    methods = sorted({r['method'] for r in rankings['queries']})
    retrieval_unit = rankings.get('retrieval_unit', 'document')
    if retrieval_unit not in {'document', 'passage'}:
        raise ValueError('Retrieval unit must be document or passage.')
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
            if retrieval_unit == 'document' and len(ranked) != len(set(ranked)):
                raise ValueError('Duplicate ranked IDs; do not silently refill top k.')
            if not set(ranked) <= universe:
                raise ValueError('Unknown retrieved ID outside the declared retrieval universe.')
            actual = row.get('actual_method') or method
            ok = row.get('status', 'completed') == 'completed' and actual == method
            complete = q.get('complete_relevance') is True
            status = ('provider_or_method_failure' if not ok else 'incomplete_judgments' if not complete
                      else 'no_eligible_relevant_items' if not relevant else 'measured')
            result = {'question_id': qid, 'method': method, 'actual_method': actual,
                      'status': status, 'relevant_count': len(relevant),
                      'retrieval_unit': retrieval_unit}
            for k in (5, 10):
                # Document-level qrels are scored by source coverage among the
                # first K retrieved passages when retrieval_unit=passage.
                hits = len(set(ranked[:k]) & relevant)
                result[f'relevant_retrieved@{k}'] = hits
                coverage = hits / len(relevant) if status == 'measured' else None
                result[f'labeled_source_coverage@{k}'] = coverage
                result[f'recall@{k}'] = coverage if full_corpus_recall_available else None
                result[f'hit@{k}'] = int(hits > 0) if status == 'measured' else None
            details.append(result)
    summaries = []
    for method in methods:
        rows = [r for r in details if r['method'] == method]
        measured = [r for r in rows if r['status'] == 'measured']
        summaries.append({'method': method, 'queries_total': len(rows), 'queries_measured': len(measured),
            'judgment_coverage': len(measured) / len(rows) if rows else None,
            'recall@5': mean(r['recall@5'] for r in measured) if measured and full_corpus_recall_available else None,
            'recall@10': mean(r['recall@10'] for r in measured) if measured and full_corpus_recall_available else None,
            'labeled_source_coverage@5': mean(r['labeled_source_coverage@5'] for r in measured) if measured else None,
            'labeled_source_coverage@10': mean(r['labeled_source_coverage@10'] for r in measured) if measured else None})
    return {'metric': 'Recall@k' if full_corpus_recall_available else 'Labeled-source coverage@k',
            'unit': qrels['unit'], 'retrieval_unit': retrieval_unit,
            'index_signature': qrels['index_signature'],
            'scope_id': qrels['scope_id'], 'scope_description': qrels.get('scope_description'),
            'judged_source_count': len(eligible), 'retrieval_universe_count': len(universe),
            'full_corpus_recall_available': full_corpus_recall_available,
            'full_corpus_recall_unavailable_reason': (
                None if full_corpus_recall_available else
                'Relevance judgments do not cover every source in the retrieval corpus.'),
            'formula': ('relevant IDs in first k retrieved passages / all relevant IDs, only when the corpus is fully judged'
                        if full_corpus_recall_available else
                        'known relevant IDs in first k retrieved passages / all currently labeled relevant IDs; not full-corpus recall'),
            'summary': summaries, 'queries': details,
            'notice': ('Recall is the share of labeled relevant sources represented in the first K retrieved '
                       'passages. The 50-source judgment pool is evaluated against the full corpus; unjudged '
                       'sources may also be relevant. This is not a full-corpus '
                       'recall guarantee, report accuracy, structural completeness or aviation risk.')}


def bm25_replay(index_dir, qrels, *, include_case_context=True):
    """Rank the full non-protected source universe using the existing index; never rebuild it."""
    from blackwing.rag.engine import lexical_rank, _is_protected_gold
    index_dir = Path(index_dir)
    manifest = json.loads((index_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['signature'] != qrels['index_signature']:
        raise ValueError('Qrels belong to a different index signature.')
    unit = qrels['unit']
    rows, represented = [], set()
    with (index_dir / 'chunks.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            identifier = row['chunk_id'] if unit == 'chunk' else row.get('source_id')
            if _is_protected_gold(row):
                continue
            rows.append(row)
            represented.add(identifier)
    if not set(qrels['eligible_ids']) <= represented:
        raise ValueError('Qrels contain IDs absent from the existing index.')
    output = {'index_signature': manifest['signature'], 'scope_id': qrels['scope_id'],
              'unit': unit, 'retrieval_unit': 'passage',
              'retrieval_universe_ids': sorted(represented), 'ranking_depth': 100,
              'ranking_rule': 'Top 100 BM25 passage hits over the full non-protected source universe',
              'queries': []}
    for q in qrels['questions']:
        ordered = []
        query = query_with_case_context(q) if include_case_context else str(q.get('query') or '')
        for _, i in lexical_rank(query, rows):
            identifier = rows[i]['chunk_id'] if unit == 'chunk' else rows[i]['source_id']
            ordered.append(identifier)
            if len(ordered) >= output['ranking_depth']:
                break
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
