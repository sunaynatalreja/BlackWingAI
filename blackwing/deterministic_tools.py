"""BLACKWING AI V2 | conservative local evidence / rule / risk checks.

This module triages retrieved text; it DOES NOT independently prove entailment,
establish a root cause, certify regulatory applicability, or calculate official risk.
No external APIs, gold answers, or model calls.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

WORD = re.compile(r"[A-Za-z0-9]+")
STOP = {
    "about", "after", "also", "and", "are", "been", "could", "from", "have",
    "into", "this", "that", "their", "there", "then", "these", "those", "were",
    "when", "where", "which", "with", "would", "was", "said", "flight", "report",
    "airport", "incident", "the", "for", "but", "not", "had", "his", "her", "its",
}


def keywords(text: str) -> set[str]:
    return {x.lower() for x in WORD.findall(str(text)) if len(x) > 3 and x.lower() not in STOP}


def build_evidence_checklist(incident: dict[str, Any], retrieved: list[dict] | None = None) -> list[dict]:
    """Describe evidence supplied to the *system*; no claim of absence in the world."""
    text = str(incident.get("narrative", ""))
    source_text = " ".join(str(r.get("text", "")) for r in (retrieved or []))
    combined = (text + " " + source_text).lower()
    checks = [
        ("event_date", ["date", "2023", "2024", "2025", "2026"]),
        ("airport_runway", ["runway", "airport", "taxiway"]),
        ("crossing_clearance", ["crossing clearance", "cross runway", "cleared to cross"]),
        ("atc_communications", ["controller", "readback", "radio", "atc", "clearance"]),
        ("aircraft_positions", ["ads-b", "asde-x", "position", "surveillance", "track"]),
        ("readback_hearback", ["readback", "hearback", "acknowledged"]),
        ("airport_diagram", ["airport diagram", "taxi route", "figure", "marking"]),
        ("investigation_conclusions", ["probable cause", "contributing factors", "final report"]),
    ]
    result = []
    for name, tokens in checks:
        found = any(token in combined for token in tokens)
        status = "MENTIONED_IN_AVAILABLE_TEXT_UNVERIFIED" if found else "NOT_IDENTIFIED_IN_AVAILABLE_TEXT"
        if name == "event_date" and incident.get("event_date"):
            status = "USER_SUPPLIED_DATE_NOT_INDEPENDENTLY_VERIFIED"
        result.append({"item": name, "status": status})
    return result


def date_aware_candidate_rules(retrieved: list[dict], event_date: str | None) -> list[dict]:
    """Exclude plainly later-dated *candidate* rules. Older is not necessarily applicable.

    Date hints from filenames are NOT proof that an edition was in force on event day.
    NTSB text quoting an order is secondary evidence for that rule, not the full text.
    """
    try:
        event = date.fromisoformat(str(event_date))
    except (ValueError, TypeError):
        event = None
    output = []
    seen = set()
    for record in retrieved:
        if record.get("source_role") != "rule_candidate":
            continue
        source_id = record.get("document_id") or record.get("filename")
        if source_id in seen:
            continue
        seen.add(source_id)
        filename = str(record.get("filename", ""))
        years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", filename)]
        publication_year_hint = min(years) if years else None
        later = bool(event and publication_year_hint and publication_year_hint > event.year)
        output.append({
            "source_id": source_id,
            "filename": filename,
            "publication_year_hint": publication_year_hint,
            "status": "EXCLUDE_POST_EVENT_DATE" if later else "EDITION_AND_BINDING_STATUS_UNVERIFIED",
            "notice": "Filename year is only a hint; verify the specific effective date, amendment and jurisdiction.",
        })
    return output


def assess_claim_support(claim: str, source_chunk: dict | None) -> dict:
    """Review aid: lexical coverage + number consistency; never logical entailment."""
    if not source_chunk or not str(source_chunk.get("text", "")).strip():
        return {"status": "NO_SELECTED_EVIDENCE_TEXT", "lexical_overlap": 0.0, "missing_numbers": []}
    text = str(source_chunk["text"])
    claimed = keywords(claim)
    present = keywords(text)
    overlap = round(len(claimed & present) / len(claimed), 3) if claimed else 0.0
    # Numbers such as times, runway IDs and flight IDs require manual checking.
    nums_claim = set(re.findall(r"\b\d[\d:.,/-]*\b", str(claim)))
    nums_source = set(re.findall(r"\b\d[\d:.,/-]*\b", text))
    missing_nums = sorted(nums_claim - nums_source)
    if missing_nums:
        status = "NUMBER_OR_TIME_NOT_PRESENT_IN_EXCERPT_REVIEW"
    elif overlap >= 0.45:
        status = "TEXTUALLY_PLAUSIBLE_MANUAL_VERIFICATION_REQUIRED"
    else:
        status = "WEAK_TEXT_OVERLAP_MANUAL_VERIFICATION_REQUIRED"
    return {
        "status": status,
        "lexical_overlap": overlap,
        "missing_numbers": missing_nums,
        "note": "Text overlap is a triage signal, not proof that this passage entails the claim.",
    }


def check_evidence(claim: str, source_chunk: dict | None) -> str:
    """Backwards-compatible short flag used by the original code."""
    return assess_claim_support(claim, source_chunk)["status"]


def screen_risk(incident: dict) -> dict:
    """Prevent ungrounded official categories / numerical probabilities."""
    return {
        "official_severity": "UNDETERMINED",
        "probability": "NOT_CALCULATED",
        "required_inputs": [
            "verified aircraft separation and geometry",
            "validated timeline and time available to react",
            "aircraft speed, takeoff/rejected-takeoff and evasive actions",
            "visibility, runway occupancy and applicable risk taxonomy",
        ],
        "review_status": "QUALIFIED_EXPERT_VALIDATION_REQUIRED",
    }


def gather_evidence_flags(stages: dict, evidence_catalog: list[dict]) -> list[dict]:
    """Find claims and check cited excerpts by ID. No gold answers used.

    Can flag wrong IDs and weak lexical overlap, not semantic falsehoods.
    """
    by_id = {r.get("chunk_id"): r for r in evidence_catalog}
    flags: list[dict] = []
    pairs = (("claim", "citation_id"), ("event", "citation_id"), ("factor", "citation_id"),
             ("procedure", "citation_id"), ("action", "citation_id"))

    def walk(obj: Any, stage: str):
        if isinstance(obj, dict):
            for claim_field, citation_field in pairs:
                claim = obj.get(claim_field)
                cited = obj.get(citation_field)
                if claim is not None and isinstance(cited, str) and cited:
                    source = by_id.get(cited)
                    check = assess_claim_support(str(claim), source)
                    flags.append({"stage": stage, "claim": str(claim)[:180],
                                  "citation_id": cited, "source_known": source is not None, **check})
                    break
            for child in obj.values():
                walk(child, stage)
        elif isinstance(obj, list):
            for child in obj:
                walk(child, stage)

    for stage_name, payload in stages.items():
        walk(payload, stage_name)
    return flags

# ---------- V3.2: structured risk/hazard audit (OFFLINE, no LLM) ----------
HAZARD_CLASSES = {"OBSERVED_HAZARD", "POTENTIAL_HAZARD", "HYPOTHESIS_FOR_REVIEW"}


def validate_hazard_register(hazards: object, evidence: list[dict],
                             direct_case_evidence: bool = True) -> dict:
    """Audit structured hazards, not their factual truth.

    Each supported hazard must cite a retrieved *case* excerpt. Hypotheses are
    permitted only when explicitly labelled and tied to an observed basis.
    This is a necessary structure check, NOT semantic entailment.
    """
    known = {r.get("chunk_id"): r for r in evidence if isinstance(r, dict)}
    issues: list[dict] = []
    if not isinstance(hazards, list):
        return {"status": "REVIEW_REQUIRED", "flags": [{"reason": "HAZARD_REGISTER_NOT_ARRAY"}],
                "hazard_count": 0, "notice": "Structural audit only; expert review required."}
    if len(hazards) > 6:
        issues.append({"reason": "HAZARD_REGISTER_EXCEEDS_SIX", "count": len(hazards)})
    if direct_case_evidence and not hazards:
        issues.append({"reason": "NO_HAZARDS_WITH_INDEXED_CASE_EVIDENCE"})
    seen = set()
    for i, hazard in enumerate(hazards, start=1):
        if not isinstance(hazard, dict):
            issues.append({"reason": "HAZARD_NOT_OBJECT", "entry": i})
            continue
        hid = str(hazard.get("hazard_id") or "").strip()
        if not hid or hid in seen:
            issues.append({"reason": "MISSING_OR_DUPLICATE_HAZARD_ID", "entry": i, "hazard_id": hid})
        seen.add(hid)
        for field in ("hazard", "observed_evidence", "potential_consequence"):
            if not str(hazard.get(field) or "").strip():
                issues.append({"reason": "MISSING_HAZARD_FIELD", "entry": i, "field": field})
        if hazard.get("classification") not in HAZARD_CLASSES:
            issues.append({"reason": "UNKNOWN_HAZARD_CLASSIFICATION", "entry": i})
        cite = hazard.get("citation_id")
        source = known.get(cite)
        if not direct_case_evidence:
            issues.append({"reason": "NO_DIRECT_CASE_EVIDENCE_FOR_HAZARD", "entry": i})
        if not source or source.get("source_role") != "evidence":
            issues.append({"reason": "HAZARD_LACKS_DIRECT_SOURCE", "entry": i, "citation_id": cite})
        if hazard.get("barrier_status") not in (None, "", "DOCUMENTED", "UNVERIFIED", "NOT_ESTABLISHED"):
            issues.append({"reason": "INVALID_BARRIER_STATUS", "entry": i})
    return {"status": "REVIEW_REQUIRED" if issues else "STRUCTURE_PASS_EXPERT_VALIDATION_REQUIRED",
            "hazard_count": len(hazards), "flags": issues,
            "notice": "Source existence and schema do not prove the hazard is factually supported."}


def validate_review_actions(actions: object, hazards: object) -> dict:
    """Check that mitigation actions are actionable and mapped to hazard IDs."""
    issues: list[dict] = []
    if not isinstance(actions, list):
        return {"status": "REVIEW_REQUIRED", "flags": [{"reason": "ACTIONS_NOT_ARRAY"}]}
    ids = {str(h.get("hazard_id")) for h in hazards if isinstance(h, dict)} if isinstance(hazards, list) else set()
    for index, action in enumerate(actions, start=1):
        if not isinstance(action, dict):
            issues.append({"reason": "ACTION_NOT_OBJECT", "entry": index})
            continue
        if not str(action.get("action") or "").strip():
            issues.append({"reason": "MISSING_SPECIFIC_ACTION", "entry": index})
        if not str(action.get("citation_id") or "").strip():
            issues.append({"reason": "ACTION_MISSING_CITATION", "entry": index})
        if ids and str(action.get("hazard_id") or "") not in ids:
            issues.append({"reason": "ACTION_HAZARD_NOT_IDENTIFIED", "entry": index,
                           "hazard_id": action.get("hazard_id")})
    return {"status": "REVIEW_REQUIRED" if issues else "STRUCTURE_PASS_EXPERT_VALIDATION_REQUIRED",
            "action_count": len(actions), "flags": issues}
