"""Create a separate IIT embedding index; never rewrite the source index."""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

# Run from the BLACKWING repository root in the VS Code terminal.
ROOT = Path.cwd().resolve()
if not (ROOT/'blackwing/rag/engine.py').is_file():
    raise RuntimeError('Run this script from the BLACKWING repository root.')
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from blackwing.rag.engine import HybridIndex, _is_protected_gold
from blackwing.rag.embeddings import make_embeddings
from blackwing.rag.tokenization import BGESubwordTokenizer, assert_budget


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', help='Call IIT embeddings and create a NEW index')
    parser.add_argument('--source', type=Path, required=True, help='Existing approved source index; chunk boundaries are reused')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing directories are refused')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or output == source or source in output.parents:
        raise RuntimeError('Choose a new output folder outside the existing index')
    snapshots = {p: digest(p) for p in source.rglob('*') if p.is_file()}
    index = HybridIndex.load(source)  # validates chunk signature; does not call models
    source_manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if any(_is_protected_gold(row) for row in index.chunks):
        raise RuntimeError('Protected gold detected in source index; abort rather than upload it')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    tokenizer = BGESubwordTokenizer()  # cached tokenizer only, no model endpoint
    for row in index.chunks:
        assert_budget(row['excerpt'], tokenizer)
    required = ('BLACKWING_EMBEDDING_URL', 'BLACKWING_EMBEDDING_API_KEY', 'BLACKWING_EMBEDDING_CA_CERT')
    missing = [name for name in required if not os.environ.get(name, '').strip()]
    ca = os.environ.get('BLACKWING_EMBEDDING_CA_CERT', '')
    if ca and not Path(ca).is_file():
        missing.append('existing trusted CA certificate file')
    url = os.environ.get('BLACKWING_EMBEDDING_URL', '')
    if url and not url.startswith('https://'):
        missing.append('HTTPS embedding URL')
    model = os.environ.get('BLACKWING_EMBEDDING_MODEL', 'bge-small-en-v1.5')
    if model not in {'bge-small-en-v1.5', 'BAAI/bge-small-en-v1.5'}:
        missing.append('verified bge-small-en-v1.5 model identifier')
    report = {'status': 'ready' if not missing else 'missing_configuration',
              'source_records': len({r.get('source_id') or r['document_id'] for r in index.chunks}),
              'chunks': len(index.chunks), 'all_token_budgets_passed': True, 'protected_gold_chunks': 0,
              'batch_size': 16, 'estimated_document_requests_before_deduplication': math.ceil(len(index.chunks)/16),
              'missing_configuration': missing, 'output': str(output), 'hosted_calls_made': 0}
    (BASE/'preflight.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if not args.build:
        return
    if missing:
        raise RuntimeError('IIT configuration incomplete; no hosted calls made')
    embeddings = make_embeddings('iit', cache_path=BASE/'cache/iit_embeddings.sqlite', tokenizer=tokenizer)
    vectors = []
    # The cache commits every batch. Rerunning resumes paid work after interruption.
    for start in range(0, len(index.chunks), 16):
        vectors.append(embeddings.encode([r['excerpt'] for r in index.chunks[start:start+16]]))
        if start % 160 == 0:
            print(f'Embedded/cached {min(start+16, len(index.chunks))}/{len(index.chunks)} chunks', flush=True)
    import numpy as np
    audit = {**index.audit, 'iit_build': {'source_signature': source_manifest['signature'],
             'existing_chunks_reused': True, 'protected_gold_excluded': True,
             'source_count_not_independent_institute_acceptance': True}}
    built = HybridIndex(index.chunks, embeddings=embeddings, vectors=np.vstack(vectors), strategy=index.strategy, audit=audit)
    if built.signature != source_manifest['signature']:
        raise RuntimeError('Citation/chunk preservation check failed')
    if any(digest(p) != h for p,h in snapshots.items()):
        raise RuntimeError('Source index changed during build')
    manifest = built.save(output)
    loaded = HybridIndex.load(output, embeddings=embeddings)
    # One additional IIT query tests compatible dense and hybrid retrieval.
    query = 'runway crossing clearance hold short aircraft conflict'
    dense = loaded.search(query, method='dense', top_k=3)['results']
    hybrid = loaded.search(query, method='hybrid', top_k=3)['results']
    if not dense or not hybrid:
        raise RuntimeError('New index saved but retrieval validation failed')
    validation = {'status': 'built_and_retrieval_validated', 'manifest': manifest,
                  'source_index_unchanged': all(digest(p)==h for p,h in snapshots.items()),
                  'dense_results': len(dense), 'hybrid_results': len(hybrid),
                  'citation_signature_preserved': True}
    (output/'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == '__main__':
    main()
