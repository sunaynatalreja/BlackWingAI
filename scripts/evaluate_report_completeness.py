"""Score existing BLACKWING exports offline; never generates or changes a report."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from blackwing.report_quality import assess_quality


def evaluate(paths, database=None):
    results, seen = [], {}
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True) if database else None
    try:
        for name in paths:
            path = Path(name)
            data = path.read_bytes()
            envelope = json.loads(data.decode("utf-8-sig"))
            report = envelope.get("report", envelope)
            if not isinstance(report, dict) or not isinstance(report.get("agent_outputs"), dict):
                raise ValueError(f"Not a BLACKWING investigation report export: {path.name}")
            identity = hashlib.sha256(json.dumps({"run_id": envelope.get("run_id"), "report": report}, sort_keys=True).encode()).hexdigest()
            if identity in seen:
                seen[identity]["duplicate_exports"].append(str(path))
                continue
            case_id = report.get("case_id") or envelope.get("case_id")
            incident = envelope.get("incident")
            provenance = "supplied incident metadata in export" if isinstance(incident, dict) else None
            if not isinstance(incident, dict) and connection and case_id:
                row = connection.execute("SELECT payload FROM records WHERE kind='cases' AND id=?", (case_id,)).fetchone()
                if row:
                    incident = json.loads(row[0])
                    provenance = "canonical case record read through SQLite mode=ro"
            entry = {"report_file": str(path), "sha256": hashlib.sha256(data).hexdigest(), "duplicate_exports": [],
                     "case_id": case_id, "run_id": envelope.get("run_id"),
                     "prompt_source": provenance or "not available; not inferred from generated report",
                     "prompt_completeness": assess_quality(incident) if isinstance(incident, dict) else None,
                     "report_completeness": assess_quality(incident or {}, report),
                     "agent_roles": sorted(report["agent_outputs"]),
                     "current_case_fact_count": len(report["agent_outputs"].get("investigation", {}).get("documented_facts", [])),
                     "critic_issue_count": len(report["agent_outputs"].get("critic", {}).get("issues", []))}
            results.append(entry)
            seen[identity] = entry
    finally:
        if connection:
            connection.close()
    return {"rubric": "downloaded BLACKWING v2 with documented schema aliases", "live_model_calls": 0,
            "input_reports_modified": False, "database_access": "read-only" if database else "not used",
            "notice": "Structural completeness only; not factual accuracy, citation entailment or aviation risk. Missing prompt metadata is not reconstructed from generated text.",
            "reports": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", nargs="+", type=Path, required=True)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protected = {p.resolve() for p in args.report}
    if args.database:
        protected.add(args.database.resolve())
    if args.output.resolve() in protected:
        parser.error("Output must be separate from input reports and the database")
    result = evaluate(args.report, args.database)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for row in result["reports"]:
        prompt = row["prompt_completeness"]
        print(json.dumps({"file": Path(row["report_file"]).name, "prompt_completeness": prompt["score"] if prompt else None,
                          "report_completeness": row["report_completeness"]["score"], "raw_report_total": row["report_completeness"]["raw_score"],
                          "flags": row["report_completeness"]["flags"]}))


if __name__ == "__main__":
    main()
