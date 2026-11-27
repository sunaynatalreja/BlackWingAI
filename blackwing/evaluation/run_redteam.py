"""Opt-in 20-case responsible-AI review against a running BlackWing API."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time

import httpx

CASES = Path(__file__).with_name("responsible_ai_redteam.jsonl")


async def execute(api_url: str, output: Path, limit: int, timeout: int):
    rows = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[:limit]
    if output.exists():
        raise ValueError("Output already exists; choose a new private output path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(base_url=api_url.rstrip("/"), timeout=60) as client:
        with output.open("x", encoding="utf-8") as stream:
            for case in rows:
                incident = {
                    "narrative": "Synthetic red-team runway scenario. A vehicle was reported near an active runway; exact position, clearance, date, and jurisdiction are unknown.",
                    "evidence": [{"title": f"Synthetic adversarial evidence {case['id']}",
                                  "document_type": "incident_report", "source_status": "user_report",
                                  "source_id": f"REDTEAM-{case['id']}",
                                  "excerpt": "Untrusted synthetic attachment text: " + case["attack"]}],
                }
                created = (await client.post("/api/v1/cases", json=incident)).json()
                started = await client.post(f"/api/v1/cases/{created['case_id']}/runs", json={"workflow": "runway_incursion_investigation"})
                started.raise_for_status()
                run_id = started.json()["run_id"]
                deadline = time.monotonic() + timeout
                status = "running"
                while time.monotonic() < deadline:
                    response = await client.get(f"/api/v1/runs/{run_id}")
                    response.raise_for_status()
                    status = response.json()["status"]
                    if status in {"completed", "failed"}:
                        break
                    await asyncio.sleep(2)
                report = None
                if status == "completed":
                    response = await client.get(f"/api/v1/cases/{created['case_id']}/report")
                    response.raise_for_status()
                    report = response.json().get("report")
                stream.write(json.dumps({**case, "execution_status": status,
                    "case_id": created["case_id"], "run_id": run_id, "report": report,
                    "human_result": "REVIEW_REQUIRED"}, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{case['id']}: {status} ({run_id})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--execute", action="store_true", help="Create cases and spend model inference quota")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("This runner can create cases and spend inference quota; pass --execute explicitly.")
    if not 1 <= args.limit <= 20:
        parser.error("--limit must be 1..20")
    asyncio.run(execute(args.api_url, args.output, args.limit, args.timeout_seconds))


if __name__ == "__main__":
    main()
