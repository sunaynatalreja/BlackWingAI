from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def run(predictions: list[dict], gold: list[dict]) -> dict:
    reference = {str(row['task_id']): row for row in gold}
    rows = []
    for pred in predictions:
        if pred.get('status') == 'PREVIEW_NOT_EXECUTED':
            continue
        name = str(pred.get('task_id', ''))
        ref = reference.get(name)
        if ref is None:
            raise ValueError(f'No approved gold anchor row for {name}')
        # Match only an actual model response; never mix preview and live outputs.
        raw = str(pred.get('raw_report') or '').casefold()
        expected = [x.strip().casefold() for x in str(ref.get('expected_phrases') or '').split('|') if x.strip()]
        forbidden = [x.strip().casefold() for x in str(ref.get('forbidden_phrases') or '').split('|') if x.strip()]
        hits = [x for x in expected if x in raw]
        bad = [x for x in forbidden if x in raw]
        rows.append({'task_id': name, 'strategy': pred['strategy'],
                     'schema_valid': pred.get('schema_valid'),
                     'gold_anchor_recall': round(len(hits) / len(expected), 4) if expected else None,
                     'forbidden_anchor_hits': len(bad), 'expected_anchors': len(expected),
                     'matched_anchors': len(hits),
                     'status': str(pred.get('status'))})
    stats = {}
    for s in ('zero_shot', 'few_shot', 'decomposition'):
        subset = [r for r in rows if r['strategy'] == s]
        vals = [r['gold_anchor_recall'] for r in subset if r['gold_anchor_recall'] is not None]
        stats[s] = {'cases':len(subset), 'schema_valid':sum(r['schema_valid'] is True for r in subset),
                    'mean_gold_anchor_recall_proxy': round(mean(vals),4) if vals else None,
                    'forbidden_anchor_hits':sum(r['forbidden_anchor_hits'] for r in subset)}
    return {'label':'DEVELOPMENT GOLD-ANCHOR PROXY — NOT FACTUAL ACCURACY', 'strategies':stats,
            'cases':rows, 'independent_report_accuracy':'NOT VERIFIED',
            'semantic_citation_accuracy':'NOT VERIFIED'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--gold',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args(argv)
    with a.predictions.open(encoding='utf8') as fh:
        preds=[json.loads(line) for line in fh if line.strip()]
    with a.gold.open(encoding='utf-8-sig',newline='') as fh:
        refs=list(csv.DictReader(fh))
    output=run(preds,refs)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x',encoding='utf8') as fh:
        json.dump(output,fh,indent=2)
    print(json.dumps(output['strategies'],indent=2))
    print('NOT verified: full factual accuracy. No human review during generation.')

if __name__=='__main__': main()
