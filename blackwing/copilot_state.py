"""Pure projections of canonical API records. This module owns no database."""
from blackwing.agents.risk.matrix import SEVERITY, LIKELIHOOD

MAX_NARRATIVE_CHARS = 30_000
REFERENCE_FRAMEWORK = "FAA AC 150/5200-37A, Figure 4-5 (provisional reference)"


def case_name(incident):
    return " · ".join(str(x) for x in (incident.get("location") or "Investigation", incident.get("incident_date"), incident.get("jurisdiction")) if x)


def risk_display(report):
    """Format backend values; never calculate or infer a risk class."""
    risk = (report or {}).get("agent_outputs", {}).get("risk") or {}
    rows = []
    for hazard in risk.get("hazards") or []:
        assessed = (hazard.get("scoring_status") == "provisional_reference_only"
                    and bool(hazard.get("risk_index")) and bool(hazard.get("risk_level")))
        rows.append({"Hazard": hazard.get("hazard_id") or "Unspecified", "Description": hazard.get("description") or "",
                     "Severity": f"{hazard['severity']} — {SEVERITY.get(hazard['severity'], 'unknown')}" if assessed and hazard.get("severity") else "Unassessed",
                     "Likelihood": f"{hazard['likelihood']} — {LIKELIHOOD.get(hazard['likelihood'], 'unknown')}" if assessed and hazard.get("likelihood") else "Unassessed",
                     "Risk index": hazard.get("risk_index") if assessed else "Unassessed",
                     "Classification": hazard.get("risk_level") if assessed else "Unassessed",
                     "Evidence IDs": ", ".join(hazard.get("evidence_ids") or []),
                     "Status": hazard.get("scoring_status") or "unassessed_insufficient_inputs"})
    return {"framework": REFERENCE_FRAMEWORK, "rows": rows,
            "assessment_status": risk.get("assessment_status") or "unassessed_insufficient_inputs",
            "notice": "Reference classification only. Jurisdictional applicability is unverified; this is not a binding FAA, DGCA, ICAO or local operational decision. Qualified human verification is required.",
            "missing_inputs": risk.get("missing_inputs") or []}


def findings_from_backend(incident, report, reviews, run_id=None):
    """Stable output paths make UI review decisions durable backend overrides."""
    rows = []
    evidence = {e["evidence_id"]: e for e in incident.get("evidence", [])}
    roles = (report or {}).get("agent_outputs") or {}
    hazard_paths = {h.get("hazard_id"): f"agent_outputs.risk.hazards.{i}" for i, h in enumerate(roles.get("risk", {}).get("hazards", []))}
    fields = (("investigation", "documented_facts", "evidence", "fact"),
              ("investigation", "timeline_events", "chronology", "event"),
              ("investigation", "reported_observations", "evidence", "reported_observation"),
              ("investigation", "hypotheses", "evidence", "hypothesis"),
              ("investigation", "contributing_factors", "evidence", "hypothesis"),
              ("risk", "hazards", "hazards", "hazard"),
              ("mitigation", "recommendations", "mitigation", "proposed_action"))
    for role, field, section, kind in fields:
        for index, item in enumerate(roles.get(role, {}).get(field) or []):
            if not isinstance(item, dict):
                continue
            path = f"agent_outputs.{role}.{field}.{index}"
            refs = item.get("evidence_ids") or []
            # Mitigations inherit only actual cited hazard references.
            if role == "mitigation":
                refs = [ref for h in roles.get("risk", {}).get("hazards", [])
                        if h.get("hazard_id") in item.get("hazard_ids", []) for ref in h.get("evidence_ids", [])]
            citations = []
            for ref in dict.fromkeys(refs):
                source = evidence.get(ref, {})
                citations.append({"source_id": source.get("source_id") or ref, "chunk_id": ref,
                                  "title": source.get("title") or ref, "excerpt": source.get("excerpt") or "",
                                  "scope": "submitted" if source else "direct",
                                  "page": source.get("page_number"), "locator": source.get("section_reference") or ""})
            row = {"finding_id": path, "case_id": incident["case_id"], "output_path": path,
                   "claim": item.get("text") or item.get("event") or item.get("description") or item.get("action") or "",
                   "kind": kind, "section": section, "status": "pending", "citations": citations,
                   "uncertainty": "Preliminary backend output; verify original sources.",
                   "event_time": item.get("time_reference") or "",
                   "potential_consequence": "", "existing_controls": item.get("existing_controls") or [],
                   "related_finding_ids": [hazard_paths[h] for h in item.get("hazard_ids", []) if h in hazard_paths]}
            for review in reviews:
                for override in review.get("overrides") or []:
                    if run_id and override.get("path") == f"runs.{run_id}.{path}" and override.get("action") in {"accept", "reject"}:
                        row.update(status="accepted" if override["action"] == "accept" else "rejected",
                                   reviewed_by=review["reviewer_id"], reviewed_at=review["created_at"],
                                   review_note=review.get("rationale") or "")
            rows.append(row)
    return rows


def report_availability(runs, report):
    """Explain report HTTP 409 using canonical run state, never provider text."""
    if report is not None:
        return None
    latest = max(runs, key=lambda row: row["created_at"]) if runs else None
    if latest and latest.get("error_code") == "interrupted":
        state = "interrupted"
        message = "Report unavailable: execution was interrupted before a report completed. Saved traces and partial outputs are preserved. Start a new controlled recovery run; this does not resume a checkpoint."
    elif latest and latest["status"] in {"queued", "running"}:
        state = "running"
        message = "Report unavailable: the investigation is still running. Refresh investigation status and inspect Run history for agent progress. Do not submit a duplicate run."
    elif latest and latest["status"] == "failed":
        state = "failed"
        message = "Report unavailable: the investigation failed before a report completed. Inspect the persisted execution trace in Run history, resolve the failure and start a new run."
    elif latest:
        state = "unavailable"
        message = "Report unavailable despite the latest run state. Refresh to reconcile backend status and inspect Run history."
    else:
        state = "not_started"
        message = "Report unavailable: no investigation has completed. Start an investigation to generate a preliminary report."
    return {"state": state, "message": message, "run_id": latest["run_id"] if latest else None}


def execution_label(progress):
    labels = {"agent_started": "Agent started", "waiting_for_model_response": "Waiting for model response",
              "agent_completed": "Agent completed", "agent_failed": "Agent failed",
              "execution_interrupted": "Execution interrupted", "execution_failed": "Investigation failed",
              "model_request_failed": "Model request failed", "model_response_received": "Model response received",
              "execution_completed": "Investigation completed"}
    event = next((e for e in reversed(progress or []) if e.get("status") in labels), None)
    if not event:
        return None
    return (event.get("agent_key") or event.get("agent") or "Investigation") + " · " + labels[event["status"]]
