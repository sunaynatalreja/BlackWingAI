"""Read approved references only for post-generation evaluation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from blackwing.gold_comparison import catalog, gold_path, match_gold


class MaterialsUnavailable(ValueError):
    """Public evaluation availability, without paths, answers or provider details."""
    def __init__(self, reason, *, reference_case=False):
        super().__init__(reason)
        self.reason = reason
        self.reference_case = reference_case


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode('utf-8')).hexdigest()


def case_materials(incident, *, manifest_path=None):
    """A final reference is never borrowed from a different incident."""
    from pypdf import PdfReader
    from blackwing.qa import workbook_rows
    item = match_gold(incident)
    questions = []
    workbook_hash = None
    if item:
        labels = {'JFK': 'JFK', 'Austin': 'AUSTIN', 'Goa Dabolim': 'GOA', 'Delhi 2013': 'DELHI2013'}
        label = labels.get(item['case_label'])
        if label is None:
            raise MaterialsUnavailable('This matching gold report has no approved evaluation questions configured.', reference_case=True)
        rows, workbook_hash = workbook_rows()
        for row in rows:
            if row.get('case_label') != label or str(row.get('human_validation_status')).upper() != 'APPROVED':
                continue
            answer = str(row.get('approved gold answer') or '').strip()
            if not answer:
                continue
            questions.append({'question_id': row['question_id'], 'question': row['query'],
                'expected_answer': answer, 'answerability': 'judge_from_input',
                'test_family': row.get('test_family'),
                'anchor_source': row.get('gold_supporting_source_ids'),
                # This locator belongs to the workbook's supporting document,
                # not necessarily to a page of the matching FINAL report.
                'anchor_source_locator': row.get('gold exact_page_or_locator')})
        path = gold_path(item)
        expected_hash = item['sha256']
        reference_id = item['gold_id']
    else:
        configured = manifest_path or os.getenv('BLACKWING_EVAL_CASES_MANIFEST')
        if not configured:
            raise MaterialsUnavailable('No matching approved question/reference case is configured.')
        manifest = json.loads(Path(configured).read_text(encoding='utf-8'))
        matches = [c for c in manifest.get('cases', [])
                   if c.get('incident_id') == incident.get('source_incident_id')
                   and c.get('incident_date') == str(incident.get('incident_date') or '')[:10]
                   and c.get('jurisdiction') == incident.get('jurisdiction')]
        if len(matches) != 1 or matches[0].get('status') != 'APPROVED':
            raise MaterialsUnavailable('No unique approved input/reference pair is configured for this incident.')
        c = matches[0]
        path = Path(c['reference_path']).resolve()
        expected_hash = c['reference_sha256']
        reference_id = c['reference_id']
        questions = c['questions']
    if not questions:
        raise MaterialsUnavailable('The reference exists, but this case has no approved evaluation questions.', reference_case=True)
    if not 1 <= len(questions) <= 30 or len({q['question_id'] for q in questions}) != len(questions):
        raise ValueError('Evaluation requires 1..30 unique approved questions.')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError('Reference hash differs; evaluation refused.')
    pages = {str(i): page.extract_text() or '' for i, page in enumerate(PdfReader(path).pages, 1)}
    return {'reference_id': reference_id, 'reference_sha256': expected_hash,
            'reference_path': str(path), 'pages': pages, 'questions': questions,
            'reference_case': True if item else None,
            'evaluation_split': 'preserved_reference_case' if item else 'held_out_reference_case',
            'workbook_sha256': workbook_hash, 'anchors_sha256': digest(questions)}


def assert_reference_unchanged(materials):
    if hashlib.sha256(Path(materials['reference_path']).read_bytes()).hexdigest() != materials['reference_sha256']:
        raise ValueError('Reference changed during evaluation.')
