#!/usr/bin/env python3
r"""BLACKWING AI: genuinely IIT-hosted BGE fixed-vs-overlap retrieval pilot.

Execute from BlackWingAI-Arun-Git project root after loading the IIT credentials:
    .\.venv\Scripts\python.exe blackwing_week2_iit_chunking_retrieval.py

Embeds both DOCUMENTS and QUERIES using IIT BGE. Builds a separate small pilot
of two official source documents; NEVER modifies the existing 27,255-vector index.
Evaluates an explicitly heuristic evidence-co-location hit@K (NOT Recall@K).
Sequential endpoint requests, modest batch sizes, persistent cache for recovery.
"""
import csv
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from blackwing_week2_chunking_pilot import CHECKS, build_units
from blackwing.rag.embeddings import IITBGE, BGESubwordTokenizer, CachedEmbeddings

INDEX = Path('rag_indexes/expanded_bge_asrs_metadata_v2')
OUT = Path('work/iit_retrieval_evaluation')
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / 'week2_iit_chunking_document_embeddings_cache.jsonl'
CSV = OUT / 'week2_iit_chunking_retrieval_comparison.csv'
JSON = OUT / 'week2_iit_chunking_retrieval_comparison.json'
BATCH = 6

GOLD_QUESTIONS = {
    'BW-G001': 'On which local date and at what airport did the AAL106/DAL1943 occurrence take place?',
    'BW-G002': 'Name the two flights and aircraft types involved in the JFK incursion.',
    'BW-G005': "What difference is shown between AAL106's assigned taxi route and the observed route?",
    'BW-G011': 'Identify the date, airport, and two flights involved in DCA23FA149.',
    'BW-G014': 'Which aircraft was arriving and which was departing, and which runway was in use?',
}


def unit_key(source, strategy, unit_id, text, identity):
    serialized = json.dumps([source, strategy, unit_id, text, identity], ensure_ascii=False)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def normalized(matrix):
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 384 or not np.isfinite(array).all():
        raise ValueError(f'IIT BGE returned invalid document/query vectors: shape={array.shape}')
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms < 1e-10):
        raise ValueError('Zero-length BGE vector')
    return array / norms


def main():
    if not (INDEX / 'chunks.jsonl').is_file():
        raise FileNotFoundError('Run from project root; index chunks not found')

    # This experiment deliberately uses only official JFK & Austin factual sources.
    # Read-only: do not load or write vectors.npy / manifest.json.
    sources = {c['source_id'] for c in CHECKS.values()}
    source_rows = defaultdict(list)
    with (INDEX / 'chunks.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get('source_id') in sources:
                    source_rows[r['source_id']].append(r)

    units = defaultdict(list)
    for source in sorted(sources):
        if not source_rows[source]:
            raise RuntimeError(f'No indexed chunks for {source}')
        per_source = build_units(source_rows[source])
        for strategy in ('fixed', 'page_adjacent_overlap2'):
            for unit_id, unit_text in per_source[strategy]:
                if unit_text.strip():
                    units[(source, strategy)].append({'id': unit_id, 'text': unit_text})

    total = sum(len(v) for v in units.values())
    if total > 260:
        raise RuntimeError(f'Pilot unexpectedly large ({total} units). No IIT requests made.')

    print('=== IIT BGE CHUNKING RETRIEVAL PILOT ===', flush=True)
    print('Sources:', ', '.join(sorted(sources)), flush=True)
    print('Document units:', total, flush=True)
    print('Maximum new IIT document embedding requests:', (total + BATCH - 1)//BATCH, flush=True)
    print('Document embedding cache:', CACHE, flush=True)
    print('Production index will NOT be modified.', flush=True)

    emb = CachedEmbeddings(IITBGE(BGESubwordTokenizer()), None)
    identity = str(emb.identity)
    if not identity.startswith('iit:'):
        raise RuntimeError(f'Refusing non-IIT embedding backend: {identity}')

    # Cache is keyed by exact unit text + source + strategy + IIT backend identity.
    previous = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if r.get('identity') == identity and len(r.get('vector', [])) == 384:
                    previous[r['key']] = r['vector']
            except (ValueError, TypeError, KeyError):
                continue

    jobs = []
    for (source, strategy), group in sorted(units.items()):
        for item in group:
            key = unit_key(source, strategy, item['id'], item['text'], identity)
            item['key'] = key
            if key not in previous:
                jobs.append((key, item['text']))
    print('Cached document units:', total - len(jobs), flush=True)
    print('New document units to embed:', len(jobs), flush=True)

    # Single active request only. Short batches prevent hitting the 8192-token limit.
    # Progress is durably saved after each successful batch for safe resume.
    for start in range(0, len(jobs), BATCH):
        batch = jobs[start:start+BATCH]
        text_batch = [txt for _, txt in batch]
        matrix = normalized(emb.encode(text_batch, query=False))
        if len(matrix) != len(batch):
            raise ValueError('IIT BGE returned wrong number of document embeddings')
        with CACHE.open('a', encoding='utf-8') as f:
            for (key, _), vec in zip(batch, matrix):
                vector = vec.astype(float).tolist()
                f.write(json.dumps({'key': key, 'identity': identity, 'vector': vector}) + '\n')
                previous[key] = vector
            f.flush()
        print(f'IIT doc embeddings: {min(start+BATCH,len(jobs))}/{len(jobs)} newly embedded', flush=True)
        time.sleep(1.1)

    question_ids = list(GOLD_QUESTIONS)
    query_matrix = normalized(emb.encode([GOLD_QUESTIONS[k] for k in question_ids], query=True))
    if len(query_matrix) != len(question_ids):
        raise ValueError('IIT BGE returned wrong number of query embeddings')
    print('IIT query embeddings:', len(query_matrix), flush=True)

    results = []
    for i, qid in enumerate(question_ids):
        spec = CHECKS[qid]
        source, case_id = spec['source_id'], spec['case_id']
        for strategy in ('fixed', 'page_adjacent_overlap2'):
            candidates = units[(source, strategy)]
            doc_matrix = normalized([previous[c['key']] for c in candidates])
            similarities = doc_matrix @ query_matrix[i]
            order = np.argsort(-similarities)
            relevant = [bool(all(re.search(p, c['text'], re.I | re.M)
                                 for p in spec['required'].values())) for c in candidates]
            ranked = [int(j) for j in order]
            matching_ranks = [rank for rank, j in enumerate(ranked, 1) if relevant[j]]
            row = {
                'question_id': qid,
                'case_id': case_id,
                'source_id': source,
                'strategy': strategy,
                'candidate_units': len(candidates),
                'heuristic_matching_units': sum(relevant),
                'first_matching_rank': min(matching_ranks) if matching_ranks else '',
                'heuristic_evidence_hit_at5': int(bool(matching_ranks and matching_ranks[0] <= 5)),
                'heuristic_evidence_hit_at10': int(bool(matching_ranks and matching_ranks[0] <= 10)),
                'top5_unit_ids': ' | '.join(candidates[j]['id'] for j in ranked[:5]),
                'embedding_backend': identity,
                'evaluation_scope': 'source_and_case_scoped_two_official_reports',
                'metric_note': 'heuristic complete-evidence hit, not Recall@K',
            }
            results.append(row)
            print(f"{qid} {strategy:23} first_evidence_rank={row['first_matching_rank'] or 'NONE':>4} "
                  f"hit@5={row['heuristic_evidence_hit_at5']} hit@10={row['heuristic_evidence_hit_at10']}", flush=True)

    with CSV.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    summary = {}
    for strategy in ('fixed', 'page_adjacent_overlap2'):
        sr = [r for r in results if r['strategy'] == strategy]
        summary[strategy] = {
            'questions': len(sr),
            'mean_heuristic_hit_at5': round(sum(r['heuristic_evidence_hit_at5'] for r in sr)/len(sr),4),
            'mean_heuristic_hit_at10': round(sum(r['heuristic_evidence_hit_at10'] for r in sr)/len(sr),4),
        }
    report = {
        'evaluation': 'IIT_BGE_fixed_vs_adjacent_overlap_source_scoped_pilot',
        'executed_utc': datetime.now(timezone.utc).isoformat(),
        'query_embedding_backend': identity,
        'document_embedding_backend': identity,
        'production_index_modified': False,
        'citation_scope': 'two official factual reports BW-EV-001 and BW-EV-006 only',
        'limitations': [
            'Only two official factual sources, selected before ranking; this is not retrieval across 777 sources.',
            'Heuristic regex conjunction defines evidence-complete units, not independently adjudicated relevance.',
            'These are evidence-hit@K diagnostics, NOT true Recall@K.',
            'Overlapping units are each independently embedded with IIT BGE and kept outside production.',
            'Does not evaluate final Qwen answer accuracy or citation correctness.',
        ],
        'summary': summary,
        'results': results,
    }
    JSON.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\n=== IIT CHUNKING RETRIEVAL COMPLETE ===', flush=True)
    for strategy, scores in summary.items():
        print(f"{strategy}: hit@5={scores['mean_heuristic_hit_at5']:.2f}, "
              f"hit@10={scores['mean_heuristic_hit_at10']:.2f}", flush=True)
    print('CSV:', CSV)
    print('JSON:', JSON)
    print('Note: neither metric is formal Recall@K; no production index changes.')


if __name__ == '__main__':
    main()
