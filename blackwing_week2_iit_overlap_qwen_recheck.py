#!/usr/bin/env python3
"""BLACKWING AI: focused IIT BGE + IIT Qwen rerun for 2 failed JFK answers.

Requirements:
  - Run from BlackWingAI-Arun-Git project root.
  - Keep blackwing_week2_chunking_pilot.py and
    blackwing_week2_iit_chunking_retrieval.py in that same directory.
  - Keep work/iit_retrieval_evaluation/week2_iit_chunking_document_embeddings_cache.jsonl.
  - Load IIT endpoint credentials using the existing secure load.ps1 script.

This is a source-scoped pilot, NOT the full 777-source RAG system or Recall@K.
Document embeddings: IIT-hosted BGE from the existing persistent cache.
Query embeddings: IIT-hosted BGE (2 fresh embeddings).
Generation: IIT-hosted Qwen3-8B (2 sequential requests).
Never edits/rebuilds the production vectors or changes the retrieval engine.
"""
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from blackwing_week2_chunking_pilot import CHECKS, build_units
from blackwing_week2_iit_chunking_retrieval import GOLD_QUESTIONS, normalized, unit_key
from blackwing.rag.embeddings import IITBGE, BGESubwordTokenizer, CachedEmbeddings

ROOT = Path('rag_indexes/expanded_bge_asrs_metadata_v2')
OUT = Path('work/iit_retrieval_evaluation')
CACHE = OUT / 'week2_iit_chunking_document_embeddings_cache.jsonl'
QUESTIONS = ('BW-G001', 'BW-G005')
TOP_K = 10
MAX_PROMPT_CHARS = 23000


def get_url():
    base = os.environ['BLACKWING_LLM_API_BASE_URL'].rstrip('/')
    return base if base.endswith('/chat/completions') else base + '/chat/completions'


def main():
    for needed in (ROOT / 'chunks.jsonl', CACHE):
        if not needed.is_file():
            raise FileNotFoundError(f'Missing {needed}. Run from project root after IIT BGE chunking pilot.')
    OUT.mkdir(parents=True, exist_ok=True)

    # Load the real IIT-backend document vectors only from prior IIT pilot cache.
    backend = CachedEmbeddings(IITBGE(BGESubwordTokenizer()), None)
    identity = str(backend.identity)
    if not identity.startswith('iit:'):
        raise RuntimeError(f'Expected IIT embedding backend, got {identity}')
    cached = {}
    for line in CACHE.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            vector = record.get('vector')
            if record.get('identity') == identity and isinstance(vector, list) and len(vector) == 384:
                cached[record['key']] = vector
        except (ValueError, KeyError, TypeError):
            continue
    if not cached:
        raise RuntimeError('No matching IIT document embedding vectors in cache.')

    target_sources = {CHECKS[qid]['source_id'] for qid in QUESTIONS}
    source_rows = {sid: [] for sid in target_sources}
    with (ROOT / 'chunks.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                sid = item.get('source_id')
                if sid in source_rows:
                    source_rows[sid].append(item)
    if any(not rows for rows in source_rows.values()):
        raise RuntimeError('Expected official JFK source missing in index')

    index = {}
    for source, rows in source_rows.items():
        units = []
        for unit_id, content in build_units(rows)['page_adjacent_overlap2']:
            key = unit_key(source, 'page_adjacent_overlap2', unit_id, content, identity)
            if key not in cached:
                raise RuntimeError('Missing IIT doc-vector cache entry: re-run earlier IIT chunking script first.')
            units.append({'unit_id': unit_id, 'text': content, 'key': key, 'source_id': source})
        index[source] = units

    # Issue one small IIT BGE query-embedding batch (no local embedding model).
    questions = [GOLD_QUESTIONS[qid] for qid in QUESTIONS]
    query_vectors = normalized(backend.encode(questions, query=True))
    if query_vectors.shape != (len(QUESTIONS), 384):
        raise RuntimeError('Unexpected IIT query embedding shape')

    url = get_url()
    key = os.environ['BLACKWING_LLM_API_KEY']
    model = os.environ['BLACKWING_LLM_API_MODEL']
    ca = os.getenv('BLACKWING_LLM_CA_CERT') or os.getenv('BLACKWING_LLM_API_CA_CERT')
    verify = ca or True
    if isinstance(verify, str) and not Path(verify).is_file():
        raise FileNotFoundError('Configured IIT certificate path was not found')

    report = []
    print('=== IIT BGE + IIT QWEN OVERLAP RECHECK ===', flush=True)
    print('Document embeddings: verified IIT cache', flush=True)
    print('Query embeddings: IIT BGE', flush=True)
    print('Model: IIT', model, flush=True)
    print('Evaluation scope: preselected official JFK source; NOT full-corpus', flush=True)

    for n, qid in enumerate(QUESTIONS):
        source = CHECKS[qid]['source_id']
        units = index[source]
        doc_vectors = normalized([cached[unit['key']] for unit in units])
        scores = doc_vectors @ query_vectors[n]
        chosen = [units[int(i)] for i in np.argsort(-scores)[:TOP_K]]

        evidence_blocks = []
        cited_ids = set()
        for rank, unit in enumerate(chosen, 1):
            # Each overlapping unit is made of up to two original indexed chunks.
            original_ids = unit['unit_id'].split(' + ')
            cited_ids.update(original_ids)
            evidence_blocks.append(
                f"Rank: {rank}\nSource: {source}\nOriginal chunk IDs: {', '.join(original_ids)}\n"
                f"Evidence:\n{unit['text']}"
            )
        assembled = '\n\n'.join(evidence_blocks)
        if len(assembled) > MAX_PROMPT_CHARS:
            raise RuntimeError(f'{qid} evidence exceeds safe prompt-size ceiling. Nothing sent.')
        # Never use the gold answer, reference patterns or expected date/taxiways in prompt.
        system = (
            'You are an aviation incident investigation assistant. '
            'Use ONLY the supplied official excerpts. Answer the requested facts clearly, '
            'not just the description of a figure. Identify the requested local event date '
            'rather than a later meeting or investigation date. '
            'For each factual statement cite the EXACT original chunk ID shown in evidence, '
            'including its :loc...:fixed... suffix. If an excerpt lacks the requested facts, '
            'say evidence is insufficient. Do not invent facts or chunk identifiers.'
        )
        user = (
            f'Incident: {CHECKS[qid]["case_id"]}\nQuestion: {GOLD_QUESTIONS[qid]}\n\n'
            f'Official evidence (source-scoped, IIT-BGE ranked):\n{assembled}'
        )
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
            'temperature': 0,
            'max_tokens': 512,
        }
        print(f'\nQUESTION {qid}: top {len(chosen)} IIT dense overlapping units', flush=True)
        response = requests.post(
            url,
            headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
            json=payload, verify=verify, timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        answer = data['choices'][0]['message'].get('content') or ''
        exact = sorted(cid for cid in cited_ids if cid in answer)
        # Heuristics below are AFTER generation; they are not included in prompt or retrieval.
        if qid == 'BW-G001':
            content_check = {
                'correct_local_date_mentioned': bool(re.search(r'\bJanuary\s+13,\s*2023\b', answer, re.I)),
                'incorrect_meeting_date_mentioned': bool(re.search(r'\bJanuary\s+31,\s*2023\b', answer, re.I)),
                'jfk_airport_mentioned': bool(re.search(r'\bJFK\b|\bJohn\s+F\.?\s+Kennedy\b', answer, re.I)),
            }
        else:
            content_check = {
                'mentions_taxiway_k': bool(re.search(r'\b(?:taxiway|TWY)\s*K\b', answer, re.I)),
                'mentions_taxiway_j': bool(re.search(r'\b(?:taxiway|TWY)\s*J\b', answer, re.I)),
                'mentions_assigned_and_observed': ('assigned' in answer.lower() and 'observed' in answer.lower()),
            }
        record = {
            'question_id': qid,
            'query_embedding_backend': identity,
            'document_embedding_backend': identity,
            'model_backend': 'IIT-hosted Qwen',
            'evaluation_scope': 'preselected_official_source_dense_top10_not_full_corpus',
            'selected_overlap_unit_ids': [u['unit_id'] for u in chosen],
            'evidence_original_chunk_ids': sorted(cited_ids),
            'answer': answer,
            'recognized_exact_chunk_citations': exact,
            'has_exact_chunk_citation': bool(exact),
            'content_heuristics': content_check,
            'usage': data.get('usage', {}),
            'citation_support_human_review': 'pending',
        }
        report.append(record)
        print('ANSWER:', answer, sep='\n', flush=True)
        print('Exact citations:', len(exact), 'Content checks:', content_check, flush=True)
        # Single active request/key limit: sequential requests (no parallelism).
        time.sleep(1.1)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = OUT / f'week2_iit_overlap_qwen_recheck_{stamp}.json'
    path.write_text(json.dumps({
        'evaluation': 'source_scoped_IIT_BGE_IIT_Qwen_fixed_failure_recheck',
        'limitations': [
            'Preselected source, not whole 777-document corpus or formal Recall@K.',
            'Ranking uses IIT-BGE dense similarity without BM25/RRF/CrossEncoder.',
            'Content checks are heuristics; a human must verify exact citation support.',
            'No gold answers were inserted into model prompts.',
        ],
        'results': report,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n=== RECHECK COMPLETE ===', flush=True)
    print('Results:', len(report), 'Saved:', path, flush=True)
    print('Production index modifications: 0', flush=True)


if __name__ == '__main__':
    main()
