#!/usr/bin/env python3
r"""BLACKWING AI Week 2: deterministic source-page chunk boundary pilot.

Usage from the BlackWingAI-Arun-Git root:
    .\.venv\Scripts\python.exe blackwing_week2_chunking_pilot.py

This evaluates whether key factual spans coexist in one context unit under
(a) saved fixed chunks, (b) same-page adjacent overlap-2, and (c) page merge.
This is a *chunking/context-coverage ablation*, not retrieval Recall@K and not
an IIT-hosted embedding benchmark. Makes ZERO model/API calls. The corpus/index
is read-only. Results are exploratory technical trial measurements.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Regex descriptions are pre-specified diagnostics, not inserted into model prompts.
CHECKS = {
    "BW-G001": {
        "case_id": "DCA23LA125", "source_id": "BW-EV-001",
        "question": "JFK local date and airport",
        "required": {
            "local_date": r"\bJanuary\s+13,\s*2023\b",
            "airport": r"\b(?:JFK|John\s+F\.?\s+Kennedy)\b",
        },
    },
    "BW-G002": {
        "case_id": "DCA23LA125", "source_id": "BW-EV-001",
        "question": "JFK two flights and aircraft types",
        "required": {
            "delta_flight": r"\b(?:DAL\s*1943|Delta\s+Air\s+Lines\s+flight\s+1943)\b",
            "american_flight": r"\b(?:AAL\s*106|American\s+Airlines\s+flight\s+106)\b",
            "boeing_737": r"\bBoeing\s+737\b",  # relaxed family match; subtype verified separately
            "boeing_777": r"\bBoeing\s+777\b",
        },
    },
    "BW-G005": {
        "case_id": "DCA23LA125", "source_id": "BW-EV-001",
        "question": "JFK assigned taxi K versus observed taxi J",
        "required": {
            "taxiway_k": r"\b(?:taxiway\s+K|TWY\s*K)\b",
            "taxiway_j": r"\b(?:taxiway\s+J|TWY\s*J)\b",
        },
    },
    "BW-G011": {
        "case_id": "DCA23FA149", "source_id": "BW-EV-006",
        "question": "Austin date, location and two flights",
        "required": {
            "local_date": r"\bFebruary\s+4,\s*2023\b",
            "austin": r"\bAustin\b",
            "southwest": r"\b(?:SWA\s*708|Southwest\s+Airlines)\b",
            "fedex": r"\b(?:FDX\s*1432|FedEx|Federal\s+Express)\b",
        },
    },
    "BW-G014": {
        "case_id": "DCA23FA149", "source_id": "BW-EV-006",
        "question": "Austin departure, arrival and runway",
        "required": {
            "southwest": r"\b(?:SWA\s*708|Southwest)\b",
            "fedex": r"\b(?:FDX\s*1432|FedEx|Federal\s+Express)\b",
            "runway_18l": r"\b(?:RWY|runway)\s*18L\b",
            "arrival_evidence": r"\b(?:approach|cleared\s+to\s+land|arriving)\b",
            "departure_evidence": r"\b(?:takeoff|departure|departing)\b",
        },
    },
}


def page_of(row):
    match = re.search(r":loc(\d+):", str(row.get("chunk_id", "")))
    if match:
        return int(match.group(1))
    for key in ("page_number", "page"):
        if row.get(key) is not None:
            return str(row[key])
    return "unknown"


def chunk_order(row):
    match = re.search(r":fixed:(\d+):", str(row.get("chunk_id", "")))
    return (int(match.group(1)) if match else 999999, str(row.get("chunk_id", "")))


def build_units(rows):
    """Return three sets of text units: fixed, adjacent-2, and full source PDF page."""
    by_page = defaultdict(list)
    for row in rows:
        by_page[(row["source_id"], page_of(row))].append(row)
    result = {"fixed": [], "page_adjacent_overlap2": [], "page_merged": []}
    for (source, page), page_rows in by_page.items():
        page_rows.sort(key=chunk_order)
        for row in page_rows:
            result["fixed"].append((row["chunk_id"], row.get("text", "")))
        for i, row in enumerate(page_rows):
            adjacent = page_rows[i : i + 2]
            result["page_adjacent_overlap2"].append((
                " + ".join(r["chunk_id"] for r in adjacent),
                "\n".join(r.get("text", "") for r in adjacent),
            ))
        result["page_merged"].append((
            f"{source}:page:{page}",
            "\n".join(r.get("text", "") for r in page_rows),
        ))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="rag_indexes/expanded_bge_asrs_metadata_v2")
    parser.add_argument("--out", default="work/iit_retrieval_evaluation")
    args = parser.parse_args()
    index_path = Path(args.index)
    chunk_file = index_path / "chunks.jsonl"
    assert chunk_file.is_file(), f"Missing {chunk_file}; run from project root"
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_ids = {c["source_id"] for c in CHECKS.values()}
    source_rows = defaultdict(list)
    all_source_pages = defaultdict(list)
    all_source_ids = set()
    count = 0
    with chunk_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            row = json.loads(line)
            if row.get("source_id"):
                all_source_ids.add(row["source_id"])
                all_source_pages[(row["source_id"], page_of(row))].append(row)
            if row.get("source_id") in source_ids:
                source_rows[row["source_id"]].append(row)
    print(f"Existing index: {count:,} chunks (read only)")
    # Corpus-wide context-size experiment: no vector rebuilding or model calls.
    corpus_lengths = {"fixed": [], "page_adjacent_overlap2": [], "page_merged": []}
    for page_rows in all_source_pages.values():
        page_rows.sort(key=chunk_order)
        lengths = [len(r.get("text", "")) for r in page_rows]
        corpus_lengths["fixed"].extend(lengths)
        corpus_lengths["page_adjacent_overlap2"].extend(
            lengths[i] + (lengths[i+1] + 1 if i+1 < len(lengths) else 0)
            for i in range(len(lengths))
        )
        corpus_lengths["page_merged"].append(sum(lengths) + max(0, len(lengths)-1))

    corpus_stats = {}
    for strategy, numbers in corpus_lengths.items():
        numbers.sort()
        n = len(numbers)
        corpus_stats[strategy] = {
            "context_units": n,
            "median_approx_tokens": (numbers[n//2] + 3)//4,
            "p95_approx_tokens": (numbers[min(n-1, int(0.95*n))] + 3)//4,
            "max_approx_tokens": (numbers[-1] + 3)//4,
            "fraction_over_1024_approx_tokens": round(sum(x > 4096 for x in numbers)/n, 4),
        }
    print("Corpus-wide source IDs:", len(all_source_ids))
    print("Corpus-wide context length comparison:")
    for strategy, st in corpus_stats.items():
        print(f"  {strategy:23} units={st['context_units']:6} "
              f"median~{st['median_approx_tokens']:4}tok "
              f"p95~{st['p95_approx_tokens']:4}tok "
              f">1024tok={st['fraction_over_1024_approx_tokens']:.1%}")
    results = []
    for qid, spec in CHECKS.items():
        candidates = [r for r in source_rows[spec["source_id"]] if r.get("case_id") == spec["case_id"]]
        if not candidates:
            raise RuntimeError(f"Missing source/case chunks for {qid}: {spec['source_id']} / {spec['case_id']}")
        strategies = build_units(candidates)
        for name, units in strategies.items():
            matches = [
                (identifier, text) for identifier, text in units
                if all(re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                       for pattern in spec["required"].values())
            ]
            min_chars = min((len(text) for _, text in matches), default=None)
            row = {
                "question_id": qid,
                "case_id": spec["case_id"],
                "source_id": spec["source_id"],
                "strategy": name,
                "context_units": len(units),
                "units_with_all_heuristic_markers": len(matches),
                "complete_context_unit_exists": int(bool(matches)),
                "min_chars_matching_unit": min_chars if min_chars is not None else "",
                "estimated_tokens_matching_unit": ((min_chars + 3) // 4) if min_chars is not None else "",
                "matched_unit_ids_sample": " | ".join(identifier for identifier, _ in matches[:3]),
            }
            results.append(row)
            print(f"{qid} | {name:23} | all-markers-in-one-unit={bool(matches)!s:5} | "
                  f"units={len(units):3} | min estimated tokens={row['estimated_tokens_matching_unit']}")
    csv_path = output_dir / "week2_chunking_boundary_pilot.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    report = {
        "test": "chunking_context_boundary_pilot",
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "index": str(index_path),
        "indexed_chunks_count": count,
        "source_ids_used": sorted(source_ids),
        "corpus_wide_source_id_count": len(all_source_ids),
        "corpus_wide_length_diagnostics": corpus_stats,
        "model_api_calls": 0,
        "iit_hosted_model_test": False,
        "review_status": "technical_trial_only_human_signoff_pending",
        "limitations": [
            "Heuristic phrase co-location, not retrieval Recall@K or answer accuracy.",
            "This is a fixed versus same-page adjacent-overlap versus page-merge ablation, not a new dense index.",
            "Evidence co-location component is limited to two factual reports and five diagnostic questions; context-length comparison includes the indexed corpus.",
            "Token counts are rough character/4 estimates, not model tokenizer measurements.",
            "Automated phrase matches can miss synonymous evidence or include irrelevant contexts.",
            "No gold answers or evaluation texts are supplied to any LLM or retriever.",
        ],
        "results": results,
    }
    json_path = output_dir / "week2_chunking_boundary_pilot.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== CHUNKING PILOT COMPLETE ===")
    print("Rows:", len(results))
    print("CSV:", csv_path)
    print("JSON:", json_path)
    print("IIT API calls: 0 (deterministic text-only chunking experiment)")


if __name__ == "__main__":
    main()
