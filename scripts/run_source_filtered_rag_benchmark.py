"""BLACKWING AI: source-filtered, read-only JFK/Austin retrieval comparison.

Run from the root of BlackWingAI-Arun-Git (using its activated .venv).
The historical reference-source hints are NOT exhaustive relevance judgments.
This script does not calculate Recall@K or claim clean held-out evaluation.
"""
from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

# Support placing this script in the repository's scripts/ directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from blackwing.rag.embeddings import CachedEmbeddings, LocalBGE
from blackwing.rag.engine import HybridIndex, CrossEncoderReranker, _is_protected_gold
from blackwing.rag.tokenization import BGESubwordTokenizer

METHODS = ("bm25", "dense", "hybrid", "rerank")
SPLITS = {"JFK": "DEVELOPMENT_10", "AUSTIN": "VALIDATION_15"}
LIMITATIONS = [
    "TECHNICAL TRIAL ONLY: underlying index sources are marked technical_trial_only.",
    "Protected BW-HO-001..009 final-report chunks are filtered before ranking and reranking; this is NOT a fully reviewed source approval.",
    "Expected source IDs are historical source hints, not exhaustive relevance judgments. Hit@K and MRR@10 are source-hint proxies, NOT Recall@K.",
    "The existing BGE index uses fixed chunks. This benchmark uses its in-memory BM25 implementation, not the reorganized separate SQLite FTS5 index.",
    "Previous 320 historical experiments used a different corpus/policy; results are NOT directly comparable as a controlled before/after estimate.",
    "Austin has been exposed during development; this is not a newly blind/held-out evaluation.",
    "Cross-Encoder can reorder candidates, but it cannot retrieve evidence absent from its candidate pool.",
]


def parse_ids(obj):
    if isinstance(obj, (list, tuple, set)):
        return {str(x).strip() for x in obj if str(x).strip()}
    if isinstance(obj, dict):
        return {str(x).strip() for x in obj.values() if isinstance(x, str) and x.strip()}
    if not obj:
        return set()
    raw = str(obj).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(raw)
            if isinstance(value, (list, tuple, set, dict)):
                return parse_ids(value)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            pass
    return {s.strip() for s in raw.replace("|", ",").replace(";", ",").split(",") if s.strip()}


def avg(data):
    return round(statistics.mean(data), 4) if data else None


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("evaluation_results/rag_benchmark"))
    p.add_argument("--rerank-candidates", type=int, default=20)
    return p.parse_args()


def load_queries(gold_path, historical_path, indexed_sources):
    with historical_path.open(encoding="utf-8") as f:
        historical = json.load(f)
    refs = {}
    for r in historical:
        if r.get("method") == "bm25":
            refs[str(r["question_id"])] = parse_ids(r.get("expected_source_ids"))

    with gold_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    questions, rejected = [], []
    seen = set()
    for row in rows:
        case = str(row.get("case_label", "")).strip().upper()
        if case not in SPLITS or row.get("evaluation_split") != SPLITS[case]:
            continue
        qid = str(row.get("question_id", "")).strip()
        if qid in seen:
            continue
        seen.add(qid)
        query = str(row.get("query", "")).strip()
        labels = refs.get(qid, set())
        usable = labels & indexed_sources
        if not query or not usable:
            rejected.append({"question_id": qid, "case": case,
                             "reason": "No query or no eligible source-hint ID matching current index",
                             "raw_expected_source_ids": sorted(labels)})
            continue
        questions.append({"question_id": qid, "case": case,
                          "query": query, "expected_source_ids": sorted(usable),
                          "excluded_unmatched_hints": sorted(labels - usable)})
    print(f"Questions with matching indexed hints: {len(questions)}; excluded: {len(rejected)}")
    print("Per case:", {case: sum(q["case"] == case for q in questions) for case in SPLITS})
    if rejected:
        print("Excluded questions (first 5):", json.dumps(rejected[:5], indent=2))
    if not questions:
        raise SystemExit("STOP: saved reference source IDs do not match this index; DO NOT report zero accuracy.")
    # Both cases must have sufficient scored examples to make a meaningful case comparison.
    if any(sum(q["case"] == case for q in questions) < 5 for case in SPLITS):
        raise SystemExit("STOP: fewer than five source-hint-scored questions for a case; inspect source-ID aliases first.")
    return questions, rejected


def summary_table(results):
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["case"], r["method"])].append(r)
    rows = []
    for (case, method), items in sorted(grouped.items()):
        times = [r["latency_seconds"] for r in items]
        rows.append({
            "case": case,
            "method": method,
            "questions_scored": len(items),
            "source_hint_hit_at_5": avg([r["source_hint_hit_at_5"] for r in items]),
            "source_hint_hit_at_10": avg([r["source_hint_hit_at_10"] for r in items]),
            "source_hint_mrr_at_10": avg([r["source_hint_mrr_at_10"] for r in items]),
            "mean_latency_seconds": avg(times),
            "median_latency_seconds": round(statistics.median(times), 4),
            "protected_hits": sum(r["protected_hits"] for r in items),
            "true_recall_at_k": "NOT MEASURED - exhaustive relevance judgments unavailable",
            "approval_status": "SOURCE_FILTERED_TECHNICAL_TRIAL_NOT_CLEAN_HELDOUT",
        })
    return rows


def write_html(path, summary, limitations, rejected, timestamp):
    headers = ["Case", "Method", "N", "Hint Hit@5", "Hint Hit@10", "Hint MRR@10", "Mean latency (s)", "Protected hits"]
    columns = ["case", "method", "questions_scored", "source_hint_hit_at_5", "source_hint_hit_at_10",
               "source_hint_mrr_at_10", "mean_latency_seconds", "protected_hits"]
    th = "".join(f"<th>{html.escape(x)}</th>" for x in headers)
    body = "\n".join("<tr>"+"".join(f"<td>{html.escape(str(row[col]))}</td>" for col in columns)+"</tr>" for row in summary)
    li = "\n".join(f"<li>{html.escape(s)}</li>" for s in limitations)
    css = "body{font-family:Segoe UI,Arial,sans-serif;background:#f6f8fb;color:#1b2d40;margin:40px;max-width:1200px}h1{color:#153d64}table{width:100%;border-collapse:collapse;background:white}th,td{padding:12px;border-bottom:1px solid #e2e8f0;text-align:left}th{background:#133958;color:white}tr:nth-child(even){background:#f1f6fa}.note{background:#fff3d5;padding:16px;border-left:5px solid #cb8a00}"
    path.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>BLACKWING AI RAG benchmark</title><style>{css}</style></head><body>'
       f'<h1>BLACKWING AI | Source-filtered RAG benchmark</h1><p>Generated: {html.escape(timestamp)}</p>'
       f'<div class="note"><strong>Technical trial, not clean held-out or true Recall@K.</strong> '
       f'Protected source identity check enforced before ranking and reranking.</div>'
       f'<h2>JFK development &amp; Austin previously exposed validation</h2><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'
       f'<h2>Methodological limitations</h2><ul>{li}</ul><p>Unscorable query count: {len(rejected)}. '
       'All reranker, source, and latency evidence is also stored in JSON.</p></body></html>', encoding="utf-8")


def write_pdf(path, summary, limitations, timestamp):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
    except ImportError as exc:
        raise SystemExit("CSV, JSON and HTML saved. To create PDF: python -m pip install reportlab ; rerun benchmark") from exc
    doc = SimpleDocTemplate(str(path), pagesize=landscape(A4), leftMargin=1.25*cm, rightMargin=1.25*cm,
                            topMargin=1.2*cm, bottomMargin=1.2*cm)
    styles = getSampleStyleSheet()
    story = [Paragraph("BLACKWING AI - RAG benchmark (technical trial)", styles['Title']),
             Paragraph(f"Generated: {xml_escape(timestamp)}", styles['Normal']), Spacer(1, 10),
             Paragraph("Source-hint rankings are NOT Recall@K. Protected Gold Reports were filtered; remaining sources need approval.", styles['Normal']),
             Spacer(1, 12)]
    table_data = [["Case", "Method", "N", "Hint Hit@5", "Hint Hit@10", "Hint MRR@10", "Latency (s)", "Protected"]]
    for row in summary:
        table_data.append([str(row[k]) for k in ["case", "method", "questions_scored", "source_hint_hit_at_5",
                                                "source_hint_hit_at_10", "source_hint_mrr_at_10", "mean_latency_seconds", "protected_hits"]])
    table = Table(table_data, repeatRows=1, hAlign='LEFT', colWidths=[2.6*cm,3.4*cm,1.3*cm,2.7*cm,2.9*cm,2.8*cm,2.8*cm,2.3*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#173f61')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),0.4,colors.lightgrey),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#eff5fa')]),
        ('FONTSIZE',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),
    ]))
    story.extend([table,Spacer(1,14),Paragraph("Limitations", styles['Heading2'])])
    story.extend(Paragraph("- "+xml_escape(line), styles['BodyText']) for line in limitations)
    doc.build(story)


def main():
    args = get_args()
    if args.rerank_candidates < 10:
        raise SystemExit("Reranker pool must be at least 10 candidates")
    for p in [args.index / 'manifest.json', args.index / 'chunks.jsonl', args.index / 'vectors.npy',args.gold,args.historical]:
        if not p.is_file():
            raise SystemExit(f"Required file not found: {p}")
    tokenizer = BGESubwordTokenizer()
    embedding = CachedEmbeddings(LocalBGE(tokenizer))
    index = HybridIndex.load(args.index, embeddings=embedding)
    print("Existing chunks loaded:", len(index.chunks))
    protected_chunks = sum(_is_protected_gold(r) for r in index.chunks)
    print("Known protected chunks in saved index:", protected_chunks)
    if not protected_chunks:
        raise SystemExit("STOP: protected Gold Report filter does not match this index")
    indexed_sources = {str(r.get('source_id')).strip() for r in index.chunks
                       if not _is_protected_gold(r) and r.get('source_id')}
    questions, rejected = load_queries(args.gold, args.historical, indexed_sources)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    target_files = [out / name for name in ['rag_benchmark_results.json','rag_benchmark_results.csv',
                                           'rag_benchmark_dashboard.html','rag_benchmark_report.pdf']]
    if any(p.exists() for p in target_files):
        raise SystemExit('STOP: result files already exist; preserve previous benchmark or choose another --out')
    reranker = CrossEncoderReranker()
    rows = []
    print('Running queries:', len(questions), 'x 4 methods. This includes real CPU cross-encoder inference.')
    for q_i, q in enumerate(questions, start=1):
        for method in METHODS:
            start = time.perf_counter()
            result = index.search(q['query'], method=method, scope='all', case_id=None, top_k=10,
                                  reranker=reranker if method=='rerank' else None,
                                  rerank_candidates=args.rerank_candidates)
            seconds = time.perf_counter() - start
            actual = result.get('retrieval_method', '')
            if method == 'rerank' and actual != 'hybrid+crossencoder':
                raise SystemExit(f"STOP: cross-encoder was not active for {q['question_id']} ({actual})")
            if method in ('dense','hybrid') and actual != method:
                raise SystemExit(f"STOP: requested {method} but got {actual}")
            hits = result.get('results', [])
            bad = [r.get('chunk_id') for r in hits if _is_protected_gold(r)]
            if bad:
                raise SystemExit(f"STOP: protected source leaked for {q['question_id']} {method}: {bad}")
            first = next((i for i,r in enumerate(hits,1) if str(r.get('source_id')) in q['expected_source_ids']), None)
            item = {
                'question_id':q['question_id'],'case':q['case'],'method':method,
                'reported_method':actual,'expected_source_ids':q['expected_source_ids'],
                'source_hint_hit_at_5':int(first is not None and first<=5),
                'source_hint_hit_at_10':int(first is not None and first<=10),
                'source_hint_mrr_at_10':round(1/first,5) if first else 0,
                'latency_seconds':round(seconds,5),'protected_hits':len(bad),
                'top10':[{'rank':i,'source_id':r.get('source_id'),'page_number':r.get('page_number'),
                          'chunk_id':r.get('chunk_id')} for i,r in enumerate(hits,1)]
            }
            rows.append(item)
        print(f"{q_i}/{len(questions)} {q['case']} {q['question_id']} completed", flush=True)
    summary = summary_table(rows)
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    report = {'title':'BLACKWING AI - source-filtered technical RAG benchmark',
              'timestamp_utc': timestamp, 'methodology':'source-hint Hit@K, MRR@10; not Recall@K',
              'index_manifest':str(args.index/'manifest.json'), 'indexed_chunk_count':len(index.chunks),
              'protected_chunks_identified':protected_chunks, 'query_count':len(questions),
              'excluded_queries':rejected, 'reranker_model':reranker.identity,
              'summary':summary, 'query_results':rows, 'limitations':LIMITATIONS}
    json_path,csv_path,html_path,pdf_path=target_files
    json_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with csv_path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0].keys()))
        writer.writeheader();writer.writerows(summary)
    write_html(html_path,summary,LIMITATIONS,rejected,timestamp)
    write_pdf(pdf_path,summary,LIMITATIONS,timestamp)
    print('\nSAVED FOUR FILES:')
    for p in target_files:
        print(p.resolve())
    print('\nCASE RESULTS:')
    for row in summary:
        print(f"{row['case']:7} {row['method']:7} N={row['questions_scored']:2} "
              f"HintHit5={row['source_hint_hit_at_5']:.3f} MRR10={row['source_hint_mrr_at_10']:.3f} "
              f"MeanLatency={row['mean_latency_seconds']:.3f}s")


if __name__=='__main__':
    main()
