"""Extract page-cited retrieval records. NO gold/final report is ever indexed.
Run from 04_CODE_STARTER: python index_pilot.py [--include-full-faa] [--include-asrs-50] [--scale]
"""
import argparse, csv, hashlib, json, re
from pathlib import Path
from itertools import islice
import pymupdf
HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = HERE.parent


def chunks(text, size=1300, overlap=180):
    # Character-based baseline. Iterate size/overlap in week-2 RAG ablation.
    words = text.split()
    if not words: return
    stride = max(50, (size-overlap)//6)  # ~6 chars/word English heuristic
    step = max(30, size//6)
    for i in range(0, len(words), stride):
        part = ' '.join(words[i:i+step])
        if len(part)>60: yield part
        if i+step>=len(words):break


def pdf_records(path, typ):
    doc_id=path.name.split('_')[0]
    if path.name.startswith('BW-REG-'): doc_id='-'.join(path.name.split('_')[0:1])
    if path.name.startswith('BW-SRC-'): doc_id=path.name[:10]  # BW-SRC-001
    if path.name.startswith('BW-REG-'): doc_id=path.name[:10]  # BW-REG-001
    if path.name.startswith('BW-NTSB-'): doc_id=path.name[:11]
    if path.name.startswith('BW-GOLD-') or 'FINAL' in str(path).upper():
        raise ValueError('Refusing to index a gold/final file: '+str(path))
    with pymupdf.open(path) as doc:
        for p in doc:
            text = p.get_text('text').strip()
            for j,chunk in enumerate(chunks(text)):
                rec={'chunk_id':f'{doc_id}-p{p.number+1:03}-c{j:02}',
                     'document_id':doc_id, 'filename':path.name,'source_type':typ,
                     'page':p.number+1,'text':chunk,'source_role':('rule_candidate' if typ=='FAA' else 'evidence'),
                     'source_status':('ADVISORY_NOT_BINDING_BY_ITSELF' if 'AC120-74B' in path.name else 'DATE_AND_CLAUSE_REVIEW' if typ=='FAA' else 'NTSB_FACTUAL')}
                yield rec


def asrs_records(path, limit, official_only=False):
    with path.open(encoding='utf-8-sig',newline='') as fh:
        selected = (row for row in csv.DictReader(fh)
                    if not official_only or (row["data_origin"] == "official_nasa_asrs_csv_export"
                                             and row["quality_flag"] == "needs_review"))
        for row in islice(selected, limit):
            aid=row['document_id']
            text=(row.get('synopsis','')+'\n'+row.get('narrative','')).strip()
            for k,part in enumerate(chunks(text)):
                yield {'chunk_id':f'{aid}-c{k:02}', 'document_id':aid,
                       'filename':path.name,'source_type':'NASA_ASRS_VOLUNTARY',
                       'page':None,'text':part,'source_role':'reporter_described',
                       'source_status':'UNVERIFIED_VOLUNTARY_REPORT',
                       'incident_month':row.get('incident_month',''),
                       'airport_code':row.get('airport_code','')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--include-full-faa',action='store_true')
    p.add_argument('--include-asrs-50',action='store_true')
    p.add_argument('--scale',action='store_true',help='Index combined 1,813 ASRS candidates instead of ASRS 50; needs mentor approval')
    p.add_argument('--official-asrs-only', action='store_true',
                   help='With --scale, include only standard-length official NASA exports')
    p.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE_ROOT)
    p.add_argument('--out', type=Path, required=True, help='Write a candidate index for review')
    a=p.parse_args()
    if a.official_asrs_only and not a.scale:
        p.error('--official-asrs-only requires --scale')
    root = a.source_root.resolve()
    out = a.out.resolve()
    if not (root / '01_JFK_PILOT' / '01_Factual_Evidence_INDEX').is_dir():
        p.error('--source-root must point to the BLACKWING_AI_REORG source project')
    if out.exists():
        p.error('--out already exists. Choose a fresh candidate output file.')
    source=[]
    source.extend((x,'NTSB_FACTUAL') for x in sorted((root/'01_JFK_PILOT'/'01_Factual_Evidence_INDEX').glob('*.pdf')))
    rules=sorted((root/'01_JFK_PILOT'/'02_FAA_2023_CANDIDATE_CHECK_CLAUSE').glob('*.pdf'))
    if a.include_full_faa:
        source.extend((x,'FAA') for x in rules)
    else:
        source.extend((x,'FAA') for x in rules if 'AC120-74B' in x.name)
    if a.scale:
        source.extend((x,'NTSB_FACTUAL') for x in sorted((root/'02_MORE_SOURCES_AFTER_PILOT'/'NTSB_OTHER_CASES_FACTUAL_INDEX_LATER').glob('*/*.pdf')))
    out.parent.mkdir(parents=True,exist_ok=True)
    count=0;docids=set()
    with out.open('x',encoding='utf-8') as fh:
        for x,kind in source:
            for r in pdf_records(x,kind):
                fh.write(json.dumps(r,ensure_ascii=False)+'\n'); count+=1;docids.add(r['document_id'])
        if a.scale:
            asrs=root/'02_MORE_SOURCES_AFTER_PILOT/ASRS_1813_CANDIDATE_CORPUS/BLACKWING_ASRS_Runway_Combined_1813.csv';lim=1813
        elif a.include_asrs_50:
            asrs=root/'01_JFK_PILOT/04_ASRS_50_OPTIONAL_REPORTED_ONLY/ASRS_Runway_Pilot_50.csv';lim=50
        else: asrs=None
        if asrs:
            for r in asrs_records(asrs, lim, a.official_asrs_only):
                fh.write(json.dumps(r,ensure_ascii=False)+'\n');count+=1;docids.add(r['document_id'])
    print(f'Indexed {count} page-labelled chunks from {len(docids)} source documents/report records -> {out}')
    print('Warning: FAA exact clause/date remains unchecked; ASRS records unverified and 1813-document count subject to mentor confirmation.')
    print('Index policy: NO NTSB final reports, no gold answers, no post-2023 FAA references for JFK 2023.')

if __name__=='__main__': main()
