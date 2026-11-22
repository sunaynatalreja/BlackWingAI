"""HTTP-only bridge from Sunayana's workspace to the existing eight-agent API.

No model gateway, RAG index, session SQLite database or independent case history.
"""
import os
from pathlib import Path
from io import BytesIO

import httpx

from blackwing.copilot_state import findings_from_backend, report_availability
from blackwing.schemas import CaseResponse, RunResponse, ReportResponse, TraceResponse, ReviewResponse


def client():
    return httpx.Client(base_url=os.getenv("BLACKWING_UI_API_URL", "http://127.0.0.1:8000").rstrip("/"),
                        timeout=float(os.getenv("BLACKWING_UI_TIMEOUT_SECONDS", "30")))


def request(method, path, *, json=None, allow_report_missing=False):
    with client() as connection:
        response = connection.request(method, path, **({"json": json} if json is not None else {}))
    if allow_report_missing and response.status_code == 409:
        return None
    response.raise_for_status()
    return response.json()


def backend_status():
    try:
        return request("GET", "/health")
    except httpx.HTTPError:
        return {"status": "unavailable"}


def list_cases():
    items = []
    offset = 0
    while True:
        payload = request("GET", f"/api/v1/cases?limit=100&offset={offset}")
        items.extend(CaseResponse.model_validate(row).model_dump(mode="json") for row in payload["items"])
        offset += len(payload["items"])
        if offset >= payload["total"] or not payload["items"]:
            return items


def file_evidence(files, provenance):
    """Convert bounded uploads to the backend's existing text evidence contract."""
    if len(files) > 30:
        raise ValueError("Attach at most 30 documents.")
    if files and not provenance.strip():
        raise ValueError("Describe the provenance of the supporting files.")
    evidence = []
    for filename, data in files:
        if len(data) > 5 * 1024 * 1024:
            raise ValueError("Each upload must be at most 5 MB.")
        suffix = Path(filename).suffix.casefold()
        if suffix == ".pdf":
            from pypdf import PdfReader
            from pypdf.errors import PyPdfError
            try:
                reader = PdfReader(BytesIO(data))
                if reader.is_encrypted:
                    raise ValueError("Encrypted PDFs are unsupported; supply accessible factual text.")
                parts = []
                total = 0
                for page in reader.pages:
                    part = page.extract_text() or ""
                    total += len(part) + 1
                    if total > 20_000:
                        raise ValueError("Extracted evidence exceeds the API's 20,000-character limit; split it into sourced excerpts.")
                    parts.append(part)
                text = "\n".join(parts)
            except (PyPdfError, NotImplementedError) as exc:
                raise ValueError("The PDF could not be read; supply an accessible PDF or text transcription.") from exc
        elif suffix == ".txt":
            text = data.decode("utf-8-sig")
        else:
            raise ValueError("Use PDF or UTF-8 text files.")
        text = text.strip()
        if not text:
            raise ValueError("No text could be extracted; supply a text transcription.")
        if len(text) > 20_000:
            raise ValueError("Extracted evidence exceeds the API's 20,000-character limit; split it into sourced excerpts.")
        if len(provenance.strip()) > 200:
            raise ValueError("Use at most 200 characters for provenance.")
        evidence.append({"title": Path(filename).name[:300], "document_type": "user_report",
                         "source_status": "user_report", "source_id": provenance.strip(), "excerpt": text})
    return evidence


def create_case(*, narrative, location="", jurisdiction="", incident_date=None, source_incident_id="", evidence=None):
    payload = {"category": "runway_incursion", "narrative": narrative, "location": location or None,
               "jurisdiction": jurisdiction or None, "incident_date": incident_date,
               "source_incident_id": source_incident_id or None, "evidence": evidence or []}
    return CaseResponse.model_validate(request("POST", "/api/v1/cases", json=payload)).model_dump(mode="json")


def add_case_evidence(case_id, evidence):
    return request("POST", f"/api/v1/cases/{case_id}/evidence", json=evidence)


def start_investigation(case_id, *, include_retrieval=True):
    return RunResponse.model_validate(request("POST", f"/api/v1/cases/{case_id}/runs", json={"include_retrieval": include_retrieval})).model_dump(mode="json")


def get_run(run_id):
    return RunResponse.model_validate(request("GET", f"/api/v1/runs/{run_id}")).model_dump(mode="json")


def get_case(case_id):
    incident = CaseResponse.model_validate(request("GET", f"/api/v1/cases/{case_id}")).model_dump(mode="json")
    runs = [RunResponse.model_validate(r).model_dump(mode="json") for r in request("GET", f"/api/v1/cases/{case_id}/runs")]
    response = request("GET", f"/api/v1/cases/{case_id}/report", allow_report_missing=True)
    envelope = ReportResponse.model_validate(response).model_dump(mode="json") if response else None
    reviews = [ReviewResponse.model_validate(r).model_dump(mode="json") for r in request("GET", f"/api/v1/cases/{case_id}/reviews")]
    report = envelope["report"] if envelope else None
    return {"incident": incident, "runs": runs, "envelope": envelope, "report": report,
            "reviews": reviews, "evidence": incident["evidence"],
            "findings": findings_from_backend(incident, report, reviews, envelope["run_id"] if envelope else None),
            "report_run_id": envelope["run_id"] if envelope else None,
            "report_availability": report_availability(runs, report)}


def get_trace(run_id):
    return TraceResponse.model_validate(request("GET", f"/api/v1/runs/{run_id}/trace")).model_dump(mode="json")


def get_audit(run_id):
    return request("GET", f"/api/v1/runs/{run_id}/audit")


def review_finding(case, finding_id, decision, reviewer, rationale):
    latest = max(case["runs"], key=lambda r: r["created_at"])
    if latest["run_id"] != case["report_run_id"] or latest["status"] != "completed":
        raise ValueError("A newer run is active or failed. Review the latest completed analysis after resolving it.")
    if finding_id not in {f["finding_id"] for f in case["findings"]}:
        raise ValueError("Select a finding from this backend report.")
    # Bind the durable override to the immutable run identity. Projection ignores
    # overrides from older runs even when output array positions are identical.
    return request("POST", f"/api/v1/cases/{case['incident']['case_id']}/reviews", json={
        "reviewer_id": reviewer, "decision": "request_changes" if decision == "reject" else "approve",
        "rationale": rationale, "overrides": [{"path": f"runs.{case['report_run_id']}.{finding_id}",
                                               "action": decision, "rationale": rationale}]})


def generate_report(case):
    from blackwing.copilot_report import compile_report
    incident = case["incident"]
    report = compile_report({**incident, "airport": incident.get("location"), "event_date": incident.get("incident_date")}, case["findings"])
    from blackwing.copilot_state import risk_display
    display = risk_display(case["report"])
    report["limitations"].append("Risk framework: " + display["framework"] + ". " + display["notice"])
    return report


def export_report(case_id, format):
    if format not in {"json", "md"}:
        raise ValueError("Unsupported export format")
    with client() as connection:
        response = connection.get(f"/api/v1/cases/{case_id}/report/export?format={format}")
    response.raise_for_status()
    return response.content
