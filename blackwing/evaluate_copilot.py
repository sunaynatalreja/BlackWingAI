"""Run case-disjoint Copilot questions without loading gold answers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import uuid

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "evaluation" / "copilot-questions.csv"
OUTPUT = HERE / "output" / "copilot_evaluation"
CASES = {
    "jfk": ("JFK-DCA23LA125-DEMO", "DEVELOPMENT_10", 1, 10),
    "austin": ("AUS-DCA23FA149", "VALIDATION_15", 11, 25),
    "goa": ("GOA-VT-BXL", "FINAL_TEST_55", 26, 40),
}
FROZEN_SETTINGS = (
    "BLACKWING_CONTEXT_LIMIT",
    "BLACKWING_INPUT_TOKEN_TARGET", "BLACKWING_MAX_OUTPUT_TOKENS", "BLACKWING_MAX_PROMPT_CHARS",
)
FACT_STATUS = "REVIEWER_ATTESTED_FACTUAL_INPUT"
RULE_STATUS = "CANDIDATE_RULE_REQUIRES_EDITION_REVIEW"


def questions(case):
    _, split, first, last = CASES[case]
    with QUESTIONS.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "question_id", "evaluation_split", "incident_id", "case_label",
            "country", "scenario_or_context", "query",
        }
        if set(reader.fieldnames or ()) != required:
            raise ValueError("Expected the answer-free question export.")
        rows = [row for row in reader if row["case_label"].casefold() == case]
    expected_ids = [f"BW-G{number:03d}" for number in range(first, last + 1)]
    actual_ids = sorted(row["question_id"] for row in rows)
    if actual_ids != expected_ids or any(
        row["evaluation_split"] != split or not row["query"].strip() for row in rows
    ):
        raise ValueError(f"Unexpected question IDs or split for {case}.")
    return sorted(rows, key=lambda row: row["question_id"])


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def json_digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def snapshot(retrieval, cap, provider=None):
    from inference import provider_name

    provider = provider_name(provider)
    endpoint_key = "BLACKWING_API_BASE_URL" if provider == "iit" else "BLACKWING_OLLAMA_URL"
    endpoint_default = "" if provider == "iit" else "http://127.0.0.1:11434"
    endpoint_digest = json_digest(os.getenv(endpoint_key, endpoint_default).strip().rstrip("/"))
    model_settings = ({"model": os.getenv("BLACKWING_MODEL", "").strip(),
                       "response_format": os.getenv("BLACKWING_IIT_RESPONSE_FORMAT", "json_schema").lower(),
                       "disable_thinking": os.getenv("BLACKWING_QWEN_NO_THINK_EXTRA_BODY", "NO")}
                      if provider == "iit" else {"model": os.getenv("BLACKWING_OLLAMA_MODEL", "qwen3:8b").strip()})
    files = [
        *HERE.glob("*.py"), HERE / "requirements.txt", HERE / "case_source_registry.json", QUESTIONS,
        *sorted((HERE / "indexes").glob("*.jsonl")), *sorted((HERE / "examples").glob("*.json")),
    ]
    return {
        "files": {path.relative_to(HERE).as_posix(): sha256(path) for path in sorted(files)},
        "provider": provider, "endpoint_sha256": endpoint_digest,
        "settings_sha256": json_digest([model_settings, {name: os.getenv(name) for name in FROZEN_SETTINGS}]),
        "retrieval": retrieval,
        "cap": cap,
    }


def goa_packet(case):
    incident = case["incident"]
    if (
        not incident.get("held_out") or incident.get("case_id") != CASES["goa"][0]
        or incident.get("jurisdiction", "").upper() not in {"INDIA", "IN"}
    ):
        raise ValueError("Expected the held-out Goa case with Indian jurisdiction.")
    documents = case["evidence"]
    if isinstance(documents, dict):
        documents = list(documents.values())
    if not documents:
        raise ValueError("Attach a reviewer-attested Goa factual packet before freezing.")
    rows = []
    for document in documents:
        role = document.get("source_role")
        status = document.get("source_status")
        expected_status = {"evidence": FACT_STATUS, "rule_candidate": RULE_STATUS}.get(role)
        filename = document.get("filename", "")
        if expected_status is None or status != expected_status:
            raise ValueError("Goa input includes a document outside the approved evidence and candidate guidance scopes.")
        if not filename or re.search(r"\b(final|gold|conclusions?)\b", filename.replace("_", " "), re.I):
            raise ValueError("Remove final-report, gold or conclusions files from Goa input.")
        digest = document.get("sha256", "")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", digest) or not document.get("provenance"):
            raise ValueError("Goa source is missing SHA-256 or source provenance.")
        if role == "evidence" and not document.get("reviewer"):
            raise ValueError("Goa factual packet is missing a reviewer.")
        rows.append({field: document.get(field) for field in (
            "document_id", "filename", "sha256", "provenance", "reviewer", "source_role", "source_status",
        )})
    if not any(row["source_role"] == "evidence" for row in rows):
        raise ValueError("Attach a reviewer-attested factual packet alongside any candidate guidance.")
    return {
        "incident_sha256": json_digest(incident),
        "documents": sorted(rows, key=lambda row: (row["sha256"], row["filename"])),
    }


def freeze(args, app):
    if not re.fullmatch(r"[a-fA-F0-9]{64}", args.approved_gold_sha256):
        raise ValueError("Provide the SHA-256 of the independently approved offline gold answer set.")
    manifest = {
        "case": "goa", "approved_gold_sha256": args.approved_gold_sha256.lower(),
        "snapshot": snapshot(args.retrieval, args.cap, args.provider),
        "packet": goa_packet(app.get_case(CASES["goa"][0])),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(f"Recorded Goa freeze manifest at {args.manifest}")


def run(args, app):
    from inference import provider_name

    provider = provider_name(args.provider)
    if provider == "iit" and not args.authorize_hosted:
        raise ValueError("Pass --authorize-hosted only after authorizing this IIT run.")
    case_id = CASES[args.case][0]
    rows = questions(args.case)
    if args.baseline and args.case != "jfk":
        raise ValueError("Prompt baseline experiments use only JFK development questions.")
    frozen = None
    if args.case == "goa":
        if not args.final_test or args.manifest is None:
            raise ValueError("Goa requires --final-test and the matching --manifest.")
        frozen = json.loads(args.manifest.read_text(encoding="utf-8"))
        if frozen.get("case") != "goa" or not re.fullmatch(
            r"[a-f0-9]{64}", frozen.get("approved_gold_sha256", "")
        ):
            raise ValueError("Manifest must identify the approved offline gold version.")
        if frozen.get("snapshot") != snapshot(args.retrieval, args.cap, args.provider):
            raise ValueError("Code, index, answer-free questions or model settings changed since freeze.")
        case = app.get_case(case_id)
        if frozen.get("packet") != goa_packet(case):
            raise ValueError("Goa case or approved evidence changed since freeze.")
        if case["turns"] or case["findings"]:
            raise ValueError("Goa already contains investigation results. Use a fresh held-out evaluation state.")
    elif args.final_test or args.manifest:
        raise ValueError("The final-test and manifest arguments apply to Goa.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_id = json_digest(frozen)[:16] if frozen else uuid.uuid4().hex[:16]
    path = OUTPUT / f"{args.case}-{run_id}.jsonl"
    with path.open("x", encoding="utf-8") as log:
        for row in rows:
            request = {
                "mode": "auto", "include_history": False, "retrieval": args.retrieval,
                "cap": args.cap, "provider": provider, "authorize_hosted": args.authorize_hosted, "cache": False,
            }
            if args.baseline:
                request["baseline"] = args.baseline
            if frozen:
                request["held_out_test"] = args.manifest
            try:
                turn = app.ask_question(case_id, row["query"], **request)
            except Exception as exc:
                turn = {"status": "runner_error", "failure_category": "tool-call error",
                        "error_type": type(exc).__name__}
            record = {
                **turn,
                "question_id": row["question_id"], "evaluation_split": row["evaluation_split"],
                "case_id": case_id, "question": row["query"],
            }
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            print(f"{row['question_id']} {record.get('status', 'unknown')}", flush=True)
    print(f"Saved {len(rows)} query executions to {path}")


def summarize(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("The evaluation log contains no query results.")
    status_counts = {}
    for item in records:
        status = item.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    elapsed = [item["latency_seconds"] for item in records if item.get("latency_seconds") is not None]
    tokens = [entry["total_tokens"] for item in records for entry in item.get("trace", [])
              if str(entry.get("step", "")).casefold() == "model" and
              isinstance(entry.get("total_tokens"), int)]
    tools = [sum(entry.get("tool_count", 1) for entry in item.get("trace", [])
                 if entry.get("step") in {"Retrieval", "Evidence checklist"} or "tool_count" in entry)
             for item in records if "trace" in item]
    summary = {
        "questions_recorded": len(records),
        "status_counts": status_counts,
        "completed_answer_rate": round(status_counts.get("completed", 0) / len(records), 4),
        "mean_latency_seconds": round(statistics.mean(elapsed), 2) if elapsed else None,
        "mean_recorded_tool_calls": round(statistics.mean(tools), 2) if tools else None,
        "recorded_model_tokens": sum(tokens) if tokens else None,
        "api_requests": sum(bool(entry.get("request_sent")) for item in records for entry in item.get("trace", [])),
    }
    print(json.dumps(summary, indent=2))


def model_options(parser):
    parser.add_argument("--provider", choices=("local", "iit"), help="Model provider; defaults to environment configuration.")
    parser.add_argument("--retrieval", choices=("Hybrid + reranker", "BM25"), default="Hybrid + reranker")
    parser.add_argument("--cap", type=int, choices=(512, 1024), default=512)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command")
    subcommands.add_parser("plan", help="Check question IDs and count cases without model calls.")
    freeze_parser = subcommands.add_parser("freeze-goa", help="Record the reviewed Goa inputs and configuration.")
    freeze_parser.add_argument("--manifest", type=Path, required=True)
    freeze_parser.add_argument("--approved-gold-sha256", required=True)
    model_options(freeze_parser)
    run_parser = subcommands.add_parser("run", help="Execute the selected case questions serially.")
    run_parser.add_argument("--case", required=True, choices=CASES)
    run_parser.add_argument("--final-test", action="store_true")
    run_parser.add_argument("--authorize-hosted", action="store_true",
                            help="Explicitly authorize this run to send evidence and requests to IIT.")
    run_parser.add_argument("--baseline", choices=("zero", "fewshot", "decomposition"),
                            help="Use one model call per JFK question for a matched prompt experiment.")
    run_parser.add_argument("--manifest", type=Path)
    model_options(run_parser)
    summary_parser = subcommands.add_parser("summarize", help="Read status, latency and token counts from a saved log.")
    summary_parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command in {None, "plan"}:
        for label, (case_id, split, _, _) in CASES.items():
            print(f"{label.upper()}: {len(questions(label))} questions | {case_id} | {split}")
        print("Delhi 2013, Delhi 2025 and synthetic rows are reserved in the broader 80-question export.")
        return
    try:
        if args.command == "summarize":
            summarize(args.path)
            return
        import blackwing_app as app
        if args.command == "freeze-goa":
            freeze(args, app)
        else:
            run(args, app)
    except (FileExistsError, KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
