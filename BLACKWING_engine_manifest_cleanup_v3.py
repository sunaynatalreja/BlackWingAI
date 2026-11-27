#!/usr/bin/env python3
r"""Surgical, safe cleanup of 3 obsolete RAG manifest generator fields.

Run from BLACKWING repo root:
  .\.venv\Scripts\python.exe .\BLACKWING_engine_manifest_cleanup_v3.py --dry-run
  .\.venv\Scripts\python.exe .\BLACKWING_engine_manifest_cleanup_v3.py

Edits ONLY blackwing/rag/engine.py. Does not modify manifest.json or any index files.
"""

from __future__ import annotations
import argparse
import ast
from datetime import datetime
import difflib
from pathlib import Path
import re
import shutil

TARGET = Path('blackwing/rag/engine.py')
KEYS = ('iit_document_count_requirement', 'true_recall_at_k', 'answer_quality')


def patch(text: str) -> tuple[str, list[str]]:
    """Remove only the verified, adjacent 4-line dictionary block."""
    ast.parse(text)
    lines = text.splitlines(keepends=True)
    match = [i for i, line in enumerate(lines)
             if re.match(r'^\s*[\'\"]iit_document_count_requirement[\'\"]\s*:', line)]
    if not match:
        # Only claim clean when there are no other references to the keys.
        existing = [key for key in KEYS if key in text]
        if existing:
            raise RuntimeError(f'Unrecognized layout for {existing}; no file changed')
        return text, []
    if len(match) != 1:
        raise RuntimeError(f'Found {len(match)} document-count field definitions; no file changed')
    i = match[0]
    if i + 3 >= len(lines):
        raise RuntimeError('Unexpected end of file; no file changed')

    # Actual layout reported by user, source lines 216–219, with a 2-line
    # conditional expression for the first dictionary entry.
    check = (
        bool(re.search(r'\"iit_document_count_requirement\"\s*:\s*\"NOT VERIFIED\"\s+if\s+not\s+self\.audit\s+else\s*$', lines[i].rstrip('\r\n'))),
        'COUNT_IN_RANGE_PENDING_IIT_ELIGIBILITY_CHECK' in lines[i+1]
          and '"NOT MET"' in lines[i+1]
          and lines[i+1].rstrip().endswith(','),
        bool(re.match(r'^\s*\"true_recall_at_k\"\s*:', lines[i+2])),
        bool(re.match(r'^\s*\"answer_quality\"\s*:', lines[i+3])),
        lines[i+2].rstrip().endswith(','),
        lines[i+3].rstrip().endswith(','),
    )
    if not all(check):
        raise RuntimeError('Expected 4-line manifest dictionary layout was not found; no file changed. '
                           'Please upload engine.py for a tailored correction.')

    new_text = ''.join(lines[:i] + lines[i+4:])
    ast.parse(new_text)
    remaining = [key for key in KEYS if key in new_text]
    if remaining:
        raise RuntimeError(f'Target field names still present elsewhere: {remaining}; no file changed')
    return new_text, lines[i:i+4]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Validate without modifying files')
    args = parser.parse_args()
    if not TARGET.is_file():
        parser.error(f'Missing {TARGET}; run from the BLACKWING repository root')
    raw = TARGET.read_bytes()
    has_bom = raw.startswith(b'\xef\xbb\xbf')
    original = raw.decode('utf-8-sig')
    updated, removed = patch(original)
    if not removed:
        print('ALREADY CLEAN: no obsolete generator fields in engine.py')
        return
    print('Identified generator dictionary entries to remove:')
    for line in removed:
        print('  ' + line.rstrip('\r\n'))
    if args.dry_run:
        print('DRY RUN PASSED: engine.py syntax valid after removing exactly 4 lines. No files changed.')
        return
    backup_dir = Path.home() / '.blackwing' / 'patch_backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f'engine_before_manifest_cleanup_v3_{datetime.now():%Y%m%d_%H%M%S_%f}.py'
    shutil.copy2(TARGET, backup)
    TARGET.write_bytes((b'\xef\xbb\xbf' if has_bom else b'') + updated.encode('utf-8'))
    if TARGET.read_bytes() != ((b'\xef\xbb\xbf' if has_bom else b'') + updated.encode('utf-8')):
        raise RuntimeError('Write verification failed; restore from backup')
    print('PATCH APPLIED: engine.py updated; Python syntax verified.')
    print('BACKUP:', backup)
    print('No JSON, vectors, BM25, RAG index, or IIT endpoints were touched.')


if __name__ == '__main__':
    main()
