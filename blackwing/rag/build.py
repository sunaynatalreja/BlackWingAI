from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .corpus import load_manifest
from .embeddings import make_embeddings
from .engine import HybridIndex
from .tokenization import ApproximateTokenizer, BGESubwordTokenizer, PLANS


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="Approved document source manifest JSON")
    p.add_argument("--root", type=Path, required=True, help="Local directory containing source documents")
    p.add_argument("--output", type=Path, required=True, help="New versioned output index folder")
    p.add_argument("--backend", choices=["none", "local", "iit"], default="none")
    p.add_argument("--strategy", choices=PLANS, default="fixed")
    p.add_argument("--compare-chunking", action="store_true", help="Record counts for all three chunk strategies")
    p.add_argument("--cache", type=Path, help="Optional SQLite embedding cache outside Git")
    p.add_argument("--require-iit-count", action="store_true", help="Fail unless 500-2000 approved unique source files")
    args = p.parse_args(argv)
    if args.output.exists():
        p.error("Output directory exists; prior evaluation/index will not be overwritten")
    records, audit = load_manifest(args.manifest, args.root)
    if not records:
        p.error("No approved, extracted source records. Check audit and manifest")
    if audit.approved_document_count > 2000:
        p.error("Track B corpus exceeds 2,000 real source files; select a smaller approved corpus")
    if args.require_iit_count and not audit.meets_iit_document_count:
        p.error("IIT Track B requires 500-2000 approved unique real source documents")
    embeddings = make_embeddings(args.backend, cache_path=args.cache)
    tokenizer = embeddings.tokenizer if embeddings else (
        BGESubwordTokenizer() if args.compare_chunking else ApproximateTokenizer())
    if args.compare_chunking:
        by_strategy = {}
        for name in PLANS:
            idx = HybridIndex.from_documents(records, strategy=name, tokenizer=tokenizer)
            by_strategy[name] = {"chunks": len(idx.chunks), "tokenizer_verified": tokenizer.verified,
                                 "retrieval_quality": "NOT VERIFIED: run separate development gold evaluation"}
        print("Chunking experiments (counts only; NOT accuracy):", json.dumps(by_strategy, indent=2))
    index = HybridIndex.from_documents(records, strategy=args.strategy, embeddings=embeddings,
                                       tokenizer=tokenizer, audit=asdict(audit))
    audit.chunks = len(index.chunks)
    index.audit = asdict(audit)
    manifest = index.save(args.output)
    # Inventory includes both excluded and accepted sources; never mistake chunks for documents.
    with args.manifest.open(encoding="utf-8-sig") as fh:
        manifest_data = json.load(fh)
    entries = manifest_data["sources"] if isinstance(manifest_data, dict) else manifest_data
    exclusions = {row["source"]: row["reason"] for row in audit.exclusions}
    indexed_ids = {row["source_id"] for row in records}
    fields = ["source_id", "path", "source_type", "document_type", "jurisdiction", "authority",
              "source_url", "publication_date", "incident_id", "incident_date", "approved",
              "indexed", "exclusion_reason"]
    with (args.output / "sources.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            writer.writerow({key: (entry.get(key, "") if key not in {"indexed", "exclusion_reason"} else
                                      (entry["source_id"] in indexed_ids if key == "indexed" else exclusions.get(entry["source_id"], "")))
                             for key in fields})
    (args.output / "audit.json").write_text(json.dumps(asdict(audit), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nBUILD COMPLETED:")
    print(json.dumps({"index": str(args.output), "manifest": manifest,
                      "indexed_by_type": dict(Counter(row["source_type"] for row in records)),
                      "source_records": len(records)}, indent=2, ensure_ascii=False))
    if not audit.meets_iit_document_count:
        print("WARNING: IIT's 500-2000 verified real documents are NOT YET demonstrated.")
    if args.backend == "none":
        print("WARNING: BM25-only index. Dense, hybrid and CrossEncoder performance NOT VERIFIED.")


if __name__ == "__main__":
    main()
