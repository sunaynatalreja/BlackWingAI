from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .corpus import GUARD

SUPPORTED = {".pdf", ".txt", ".md", ".json", ".jsonl", ".csv"}
EXCLUDE_DIRS = {".venv", "venv", ".git", "__pycache__", "rag_runs", "indexes", "evaluations", "outputs"}


def draft_manifest(root: Path) -> dict:
    root = root.resolve()
    sources = []
    seen_sha = {}
    excluded = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        relative = path.relative_to(root).as_posix()
        if any(part.casefold() in EXCLUDE_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        if GUARD.search(relative):
            excluded.append({"path": relative, "reason": "protected or evaluation file"})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen_sha:
            excluded.append({"path": relative, "reason": f"duplicate of {seen_sha[digest]}"})
            continue
        seen_sha[digest] = relative
        ident = "SRC-" + digest[:12].upper()
        sources.append({"source_id": ident, "path": relative,
                        "approved": False, "source_type": "REVIEW_REQUIRED",
                        "document_type": "REVIEW_REQUIRED", "authority": "",
                        "jurisdiction": "", "publication_date": "", "source_url": "",
                        "content_hash_sha256": digest,
                        "review_note": "Verify provenance, dates, source type, evaluation split and PII before approval"})
    return {"sources": sources, "excluded": excluded,
            "audit": {"candidate_source_files": len(sources), "unique_approved_documents": 0,
                      "iit_document_count_status": "NOT VERIFIED"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Refusing to overwrite existing manifest")
    result = draft_manifest(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Candidate files: {result['audit']['candidate_source_files']}; all set approved=false")
    print(f"Review manifest: {args.output}")


if __name__ == "__main__":
    main()
