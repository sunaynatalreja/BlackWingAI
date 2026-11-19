r"""BLACKWING AI | score FIVE reviewed questions using saved benchmark JSON.

NO inference, index rebuild, Ollama, or IIT API needed.

Run in repo root (PowerShell):
  python .\evaluation_results\rag_benchmark\blackwing_5q_score_source_recall.py `
    --index "$bge" `
    --benchmark ".\evaluation_results\rag_benchmark\rag_benchmark_results.json" `
    --labels ".\evaluation_results\rag_benchmark\blackwing_5q_document_relevance_labels.csv" `
    --out ".\evaluation_results\rag_benchmark"

Important: metric = fraction of all labeled relevant SOURCE DOCUMENTS that occur
within the FIRST K PASSAGE (CHUNK) hits. Not exhaustive chunk Recall@K.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

TARGETS={'BW-G001','BW-G002','BW-G005','BW-G011','BW-G014'}
METHODS={'bm25','dense','hybrid','rerank'}
PROTECTED=re.compile(r'(?<![A-Za-z0-9])BW-HO-00[1-9](?![0-9])',re.I)


def load_csv(path:Path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def load_index_source_ids(path:Path):
    chunks_file=path/'chunks.jsonl'
    if not chunks_file.is_file():raise SystemExit('INDEX NOT FOUND: '+str(chunks_file))
    sources=set();rows=0
    with chunks_file.open('r',encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            record=json.loads(line)
            sid=str(record.get('source_id') or '').strip()
            if not sid:raise SystemExit('STOP: index chunk missing source_id. Refuse to claim source-level recall.')
            sources.add(sid);rows+=1
    return sources,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True,help='global_v3_blind_20261118 folder')
    p.add_argument('--benchmark',type=Path,required=True,help='existing rag_benchmark_results.json')
    p.add_argument('--labels',type=Path,required=True,help='downloaded five-question document labels CSV')
    p.add_argument('--out',type=Path,default=Path('evaluation_results/rag_benchmark'))
    args=p.parse_args()
    index_sources,nchunks=load_index_source_ids(args.index)
    protected={s for s in index_sources if PROTECTED.search(s)}
    eligible=index_sources-protected
    label_rows=load_csv(args.labels)
    labels={(str(r['question_id']),str(r['source_id'])):r for r in label_rows}
    all_label_sources={r['source_id'] for r in label_rows}

    absent=eligible-all_label_sources
    if absent:
        raise SystemExit('STOP: ' + str(len(absent)) + ' BGE-indexed sources have no relevance judgment: '+repr(sorted(absent))+'\nThis is not a complete label set for your global BGE corpus. Share index source IDs for an update; do not report true Recall.')
    missing=[(q,s) for q in sorted(TARGETS) for s in sorted(eligible) if (q,s) not in labels]
    if missing:raise SystemExit('STOP: missing some question/source labels: '+repr(missing[:20]))
    questions={r['question_id'] for r in label_rows}
    if not TARGETS <= questions:raise SystemExit('STOP: labels missing required five questions')
    gold={q:{s for s in eligible if int(labels[(q,s)]['binary_relevant'])==1} for q in TARGETS}
    if any(not r for r in gold.values()):raise SystemExit('STOP: missing positive relevant docs in evaluated index')

    with args.benchmark.open(encoding='utf-8') as f: benchmark=json.load(f)
    results=benchmark.get('query_results',[])
    selected=[r for r in results if r.get('question_id') in TARGETS]
    expected={(q,m) for q in TARGETS for m in METHODS}
    got=[(r.get('question_id'),r.get('method')) for r in selected]
    if len(got)!=20 or set(got)!=expected or len(set(got))!=20:
        raise SystemExit(f'STOP: benchmark must contain all 5x4 saved evaluations; got {len(got)} items, {len(set(got))} unique pairs.')

    details=[]
    for result in selected:
        qid=result['question_id'];method=result['method'];hits=result['top10']
        if len(hits)<10:raise SystemExit(f'STOP: fewer than 10 ranked passages for {qid}/{method}')
        if any(PROTECTED.search(str(h.get('source_id') or '')) for h in hits):
            raise SystemExit(f'STOP: protected report in stored ranked hits for {qid}/{method}')
        if any(str(h.get('source_id')) not in eligible for h in hits):
            raise SystemExit(f'STOP: hit source missing from the indexed 50-document eligible source set for {qid}/{method}')
        actual5={str(h.get('source_id')) for h in hits[:5]}
        actual10={str(h.get('source_id')) for h in hits[:10]}
        relevant=gold[qid]
        details.append({
            'question_id':qid,'case':result.get('case'), 'method':method,
            'relevant_source_count':len(relevant),
            'relevant_source_ids':'; '.join(sorted(relevant)),
            'retrieved_unique_sources_in_first_5_passages':len(actual5),
            'retrieved_unique_sources_in_first_10_passages':len(actual10),
            'relevant_sources_hit_first_5_passages':'; '.join(sorted(relevant & actual5)),
            'relevant_sources_hit_first_10_passages':'; '.join(sorted(relevant & actual10)),
            'source_recall_in_top_5_passages':round(len(relevant & actual5)/len(relevant),6),
            'source_recall_in_top_10_passages':round(len(relevant & actual10)/len(relevant),6),
        })
    agg=defaultdict(list)
    for r in details:
        agg[(r['case'],r['method'])].append(r)
        agg[('COMBINED',r['method'])].append(r)
    summary=[]
    for (case,method),group in sorted(agg.items()):
        summary.append({'case':case,'method':method,'n_questions':len(group),
          'mean_source_recall_in_top_5_passages':round(statistics.mean(r['source_recall_in_top_5_passages'] for r in group),6),
          'mean_source_recall_in_top_10_passages':round(statistics.mean(r['source_recall_in_top_10_passages'] for r in group),6),
          'label_scope':'Document-level judgments over local BGE indexed source IDs; among first K passage results.',
          'review_status':'ANALYST_ASSISTED_DRAFT_FOR_HUMAN_SIGNOFF',
        })
    output=args.out;output.mkdir(parents=True,exist_ok=True)
    for filename,records in [('blackwing_5q_source_recall_details.csv',details),('blackwing_5q_source_recall_summary.csv',summary)]:
        with (output/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    report={
        'title':'BLACKWING AI: 5-question SOURCE DOCUMENT coverage in top K retrieved passages',
        'classification':'TECHNICAL_TRIAL_ANALYST_ASSISTED_DRAFT',
        'benchmark_source':str(args.benchmark),'label_source':str(args.labels),
        'index_chunk_count':nchunks,'index_source_count':len(index_sources),
        'protected_source_count':len(protected),'nonprotected_index_source_count':len(eligible),
        'all_nonprotected_index_sources_human_judgment_label_present':True,
        'recall_definition':'# distinct relevant source IDs represented in top K chunk hits / # all judged relevant source IDs in indexed corpus',
        'not_claimed':'Not chunk-level Recall@K, not recall at K unique ranked documents, not approved gold without human review.',
        'selected_questions':sorted(TARGETS),
        'relevance_definition':'Grades 1 (partial useful) and 2 (direct complete) count as relevant.',
        'summary':summary,'details':details}
    (output/'blackwing_5q_source_recall_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'INDEX: {nchunks} chunks; {len(index_sources)} source IDs; {len(protected)} protected; {len(eligible)} nonprotected')
    print('COMPLETE LABEL COVERAGE FOR INDEX: YES')
    print('SOURCE-LEVEL RECALL AMONG TOP K PASSAGE HITS (not chunk Recall@K):')
    for r in summary:
        print(f"{r['case']:8} {r['method']:7} N={r['n_questions']}  @5={r['mean_source_recall_in_top_5_passages']:.3f}  @10={r['mean_source_recall_in_top_10_passages']:.3f}")
    print('SAVED: three new recall files in',output.resolve())

if __name__=='__main__':main()
