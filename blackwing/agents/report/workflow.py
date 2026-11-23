"""Task-decomposed report drafting over the completed agent outputs."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import httpx

from blackwing.agents.base import AgentSpec
from blackwing.agents.runtime import invoke_agent
from blackwing.config import Settings
from .schema import (
    REPORT_TASK_ORDER,
    ReportAgentOutput,
    ReportSection,
    ReportSectionDraft,
)

logger = logging.getLogger("blackwing.agents.report")
_PACKAGE = Path(__file__).parent
_PROMPT_SECTION_HEADINGS = {
    "KNOWN INPUT LIMITS", "SUPPLIED SYNTHETIC ACCOUNTS", "INVESTIGATION REQUEST",
    "CASE-SPECIFIC VERIFICATION FOCUS", "VERIFICATION FOCUS",
}

_TASKS = {
    "overview": {
        "title": "Incident overview",
        "prompt": "overview.md",
        "sources": ("investigation", "verification"),
    },
    "findings_and_causes": {
        "title": "Findings and causal assessment",
        "prompt": "findings_and_causes.md",
        "sources": ("investigation", "critic", "verification"),
    },
    "comparisons_and_compliance": {
        "title": "Historical comparisons and compliance",
        "prompt": "comparisons_and_compliance.md",
        "sources": ("similar_case", "compliance"),
    },
    "risk_and_follow_up": {
        "title": "Risk and recommended follow-up",
        "prompt": "risk_and_follow_up.md",
        "sources": ("risk", "mitigation", "critic", "verification"),
    },
}


def _task_source_outputs(task_id: str, prior: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {key: prior[key] for key in _TASKS[task_id]["sources"] if key in prior}


def _collect_allowed_references(value: Any) -> tuple[set[str], set[str]]:
    evidence: set[str] = set()
    sources: set[str] = set()
    if isinstance(value, dict):
        for key in ("evidence_ids", "chunk_ids"):
            entries = value.get(key)
            if isinstance(entries, list):
                evidence.update(str(item) for item in entries if item)
        for key in ("evidence_id", "chunk_id"):
            item = value.get(key)
            if item:
                evidence.add(str(item))
        for key in ("source_ids",):
            entries = value.get(key)
            if isinstance(entries, list):
                sources.update(str(item) for item in entries if item)
        for key in ("source_id", "document_id"):
            item = value.get(key)
            if item:
                sources.add(str(item))
        for child in value.values():
            child_evidence, child_sources = _collect_allowed_references(child)
            evidence.update(child_evidence)
            sources.update(child_sources)
    elif isinstance(value, list):
        for child in value:
            child_evidence, child_sources = _collect_allowed_references(child)
            evidence.update(child_evidence)
            sources.update(child_sources)
    return evidence, sources


def _ground_sections(sections: list[dict[str, Any]], prior: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    allowed_evidence, allowed_sources = _collect_allowed_references(prior)
    for section in sections:
        for point in section.get("key_points", []):
            point["evidence_ids"] = list(dict.fromkeys(
                item for item in point.get("evidence_ids", []) if item in allowed_evidence
            ))
            point["source_ids"] = list(dict.fromkeys(
                item for item in point.get("source_ids", [])
                if item in allowed_sources and item not in point["evidence_ids"]
            ))
            if point.get("classification") in {"documented_fact", "reported_observation"} and not (
                point["evidence_ids"] or point["source_ids"]
            ):
                point["classification"] = "verification_gap"
    return sections


def _claim_points(rows: Any, default_classification: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    result = []
    state_map = {
        "documented_fact": "documented_fact",
        "reported_observation": "reported_observation",
        "hypothesis": "hypothesis",
        "analytical_inference": "hypothesis",
        "unknown": "verification_gap",
    }
    for row in rows:
        if isinstance(row, str):
            text = row
            state = default_classification
            evidence_ids = []
            source_ids = []
        elif isinstance(row, dict):
            text = row.get("text") or row.get("description") or row.get("event") or row.get("rationale")
            if row.get("event") and row.get("time_reference"):
                text = str(row["time_reference"]) + ": " + str(text)
            state = row.get("evidence_state")
            evidence_ids = row.get("evidence_ids", [])
            source_ids = row.get("source_ids", []) or ([row["source_id"]] if row.get("source_id") else [])
        else:
            continue
        if not text:
            continue
        result.append({
            "text": str(text),
            "classification": state_map.get(str(state or ""), default_classification),
            "evidence_ids": evidence_ids,
            "source_ids": source_ids,
        })
    return result


def _requested_verification_focus(narrative: str) -> str | None:
    """Retain an explicitly labeled case verification focus as a follow-up, not a fact."""
    lines = narrative.splitlines()
    headings = {"CASE-SPECIFIC VERIFICATION FOCUS", "VERIFICATION FOCUS"}
    for index, line in enumerate(lines):
        if line.strip().rstrip(":").upper() not in headings:
            continue
        for candidate in lines[index + 1:]:
            value = candidate.strip()
            if value:
                if value.startswith("="):
                    return None
                return value
    return None


def _prompt_case_context(narrative: str) -> tuple[str | None, str | None]:
    """Extract the explicitly labeled scenario paragraph and input limits."""
    lines = narrative.splitlines()
    scenario = None
    for index, line in enumerate(lines):
        if not line.strip().upper().startswith("CASE ID:"):
            continue
        content = []
        for candidate in lines[index + 1:]:
            value = candidate.strip()
            if not value or value.upper() in _PROMPT_SECTION_HEADINGS or value.startswith("="):
                break
            content.append(value)
        if content:
            scenario = " ".join(content)
        break
    limits = None
    for index, line in enumerate(lines):
        if line.strip().upper() != "KNOWN INPUT LIMITS":
            continue
        content = []
        for candidate in lines[index + 1:]:
            value = candidate.strip()
            if not value or value.upper() in _PROMPT_SECTION_HEADINGS or value.startswith("="):
                break
            content.append(value)
        if content:
            limits = " ".join(content)
        break
    return scenario, limits


def _fallback_section(task_id: str, prior: Mapping[str, dict[str, Any]],
                      evidence: list[dict[str, Any]] | None = None,
                      narrative: str = "") -> dict[str, Any]:
    investigation = prior.get("investigation") or {}
    similar = prior.get("similar_case") or {}
    compliance = prior.get("compliance") or {}
    risk = prior.get("risk") or {}
    mitigation = prior.get("mitigation") or {}
    critic = prior.get("critic") or {}
    verification = prior.get("verification") or {}
    points: list[dict[str, Any]] = []

    if task_id == "overview":
        summary = investigation.get("incident_summary") or "An incident narrative was submitted for review."
        points += _claim_points(investigation.get("timeline_events"), "reported_observation")
        points += _claim_points(investigation.get("documented_facts"), "verification_gap")
        points += _claim_points(investigation.get("reported_observations"), "reported_observation")
        # Carry each submitted account into the visible report even when a
        # small model omits it from its bounded investigation JSON.
        for row in evidence or []:
            evidence_id = str(row.get("evidence_id") or row.get("chunk_id") or "")
            if (row.get("source_type") != "user_submitted"
                    or row.get("evidence_role") not in {"reported", "submitted"}
                    or evidence_id.startswith("CASE-NARRATIVE:")):
                continue
            text = row.get("excerpt") or row.get("text")
            if not evidence_id or not text:
                continue
            points.append({
                "text": str(text),
                "classification": "reported_observation",
                "evidence_ids": [evidence_id],
                "source_ids": ([str(row["source_id"])] if row.get("source_id")
                               and str(row["source_id"]) != evidence_id else []),
            })
        scenario, limits = _prompt_case_context(narrative)
        narrative_id = next((str(row.get("evidence_id") or row.get("chunk_id"))
                             for row in evidence or []
                             if str(row.get("evidence_id") or row.get("chunk_id") or "").startswith("CASE-NARRATIVE:")), None)
        if scenario:
            points.append({
                "text": "Submitted scenario context: " + scenario,
                "classification": "reported_observation",
                "evidence_ids": [narrative_id] if narrative_id else [],
                "source_ids": [],
            })
        if limits:
            points.append({
                "text": "Stated intake limits: " + limits,
                "classification": "verification_gap",
                "evidence_ids": [narrative_id] if narrative_id else [],
                "source_ids": [],
            })
    elif task_id == "findings_and_causes":
        supplied_accounts = [
            str(row.get("excerpt") or row.get("text") or "").strip()
            for row in evidence or []
            if row.get("source_type") == "user_submitted"
            and row.get("evidence_role") in {"reported", "submitted"}
            and not str(row.get("evidence_id") or row.get("chunk_id") or "").startswith("CASE-NARRATIVE:")
            and (row.get("excerpt") or row.get("text"))
        ]
        if supplied_accounts:
            summary = (
                "The supplied accounts state: " + " ".join(supplied_accounts)
                + " Their relationship cannot be resolved from unverified statements alone; "
                  "the record does not establish runway entry or probable cause."
            )
        else:
            summary = "The agent review does not establish a probable cause. Candidate explanations remain provisional until primary evidence is verified."
        # A useful findings section starts with what the evidence actually says.
        # The previous fallback emitted only unknowns, which made a populated
        # investigation look empty when a report model response failed schema validation.
        points += _claim_points(investigation.get("timeline_events"), "reported_observation")
        points += _claim_points(investigation.get("documented_facts"), "documented_fact")
        points += _claim_points(investigation.get("reported_observations"), "reported_observation")
        points += _claim_points(investigation.get("hypotheses"), "hypothesis")
        points += _claim_points(investigation.get("contributing_factors"), "hypothesis")
        for row in evidence or []:
            evidence_id = str(row.get("evidence_id") or row.get("chunk_id") or "")
            if (row.get("source_type") == "user_submitted"
                    and row.get("evidence_role") in {"reported", "submitted"}
                    and evidence_id and not evidence_id.startswith("CASE-NARRATIVE:")
                    and (row.get("excerpt") or row.get("text"))):
                points.append({
                    "text": str(row.get("excerpt") or row.get("text")),
                    "classification": "reported_observation",
                    "evidence_ids": [evidence_id],
                    "source_ids": [],
                })
        points += _claim_points(verification.get("claims_needing_source"), "verification_gap")
        points += _claim_points(investigation.get("unknowns"), "verification_gap")
    elif task_id == "comparisons_and_compliance":
        summary = str(similar.get("assessment") or "Historical comparisons are analogies only.")
        summary += " Compliance status: " + str(compliance.get("overall_applicability") or "not assessed") + "."
        for match in similar.get("matches", []):
            if isinstance(match, dict):
                points.append({
                    "text": f"{match.get('title') or 'Historical case'}: {match.get('similarity') or 'comparison requires review'}",
                    "classification": "historical_analogy",
                    "evidence_ids": [],
                    "source_ids": [str(match["source_id"])] if match.get("source_id") else [],
                })
        points += _claim_points(compliance.get("checks"), "verification_gap")
        points += _claim_points(compliance.get("missing_context"), "verification_gap")
    else:
        indices = [str(h["risk_index"]) + " (" + str(h.get("risk_level")) + ")"
                   for h in risk.get("hazards", [])
                   if h.get("scoring_status") == "provisional_reference_only" and h.get("risk_index")]
        candidate_hazards = [str(h.get("description") or "").strip()
                             for h in risk.get("hazards", []) if str(h.get("description") or "").strip()]
        scoring_reasons = list(dict.fromkeys(str(h.get("scoring_reason") or "").strip()
                             for h in risk.get("hazards", []) if str(h.get("scoring_reason") or "").strip()))
        assessment = ("Reference risk index: " + ", ".join(indices) + ". " if indices else
                      ("Potential hazard(s) identified for human verification: " + "; ".join(candidate_hazards) +
                       ". Risk not scored: " + (" ".join(scoring_reasons) if scoring_reasons else
                       "severity and likelihood evidence is unavailable") + ". "
                       if candidate_hazards else
                       "No candidate hazard was established from the supplied evidence; severity and likelihood remain unassessed. "))
        summary = assessment + str(risk.get("risk_notes") or "No rating or cause is assigned pending evidence review.")
        points += _claim_points(risk.get("hazards"), "hypothesis")
        for hazard in risk.get("hazards", []):
            if not isinstance(hazard, dict) or hazard.get("rating_basis_type") != "historical_analogies":
                continue
            rating_ids = list(dict.fromkeys(hazard.get("rating_evidence_ids") or []))
            for label, field in (("Severity basis", "severity_basis"), ("Likelihood basis", "likelihood_basis")):
                basis = str(hazard.get(field) or "").strip()
                if basis:
                    points.append({
                        "text": f"{label} (analogy-informed, provisional): {basis}",
                        "classification": "historical_analogy",
                        "evidence_ids": [],
                        "source_ids": rating_ids,
                    })
        for recommendation in mitigation.get("recommendations", []):
            if isinstance(recommendation, dict) and recommendation.get("action"):
                points.append({
                    "text": str(recommendation["action"]), "classification": "proposal",
                    "evidence_ids": [], "source_ids": [],
                })
        # Keep this section actionable and readable: critic issue lists often
        # repeat the same broad gaps several times. Prefer the verification
        # agent's concrete open questions and only retain distinct conflicts.
        points += _claim_points(verification.get("claims_needing_source"), "verification_gap")
        points += _claim_points(verification.get("conflicts"), "verification_gap")
        focus = _requested_verification_focus(narrative)
        if focus:
            if focus not in str(summary):
                summary += " Follow-up: " + focus
            points.append({
                "text": "Requested verification focus: " + focus,
                "classification": "verification_gap",
                "evidence_ids": [],
                "source_ids": [],
            })

    return {
        "task_id": task_id,
        "summary": str(summary),
        "key_points": points,
        "generation_method": "deterministic_fallback",
    }


def _merge_source_detail(section: dict[str, Any], task_id: str,
                         prior: Mapping[str, dict[str, Any]],
                         evidence: list[dict[str, Any]] | None = None,
                         narrative: str = "") -> dict[str, Any]:
    """Restore source-agent details a short-model draft may have omitted.

    These points are copied from already validated agent outputs; this step
    does not invent or upgrade claims. The report schema allows ten combined
    source and model points.
    """
    source_points = _fallback_section(task_id, prior, evidence, narrative).get("key_points", [])
    combined: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    # Source details lead so they remain visible if a model draft is repetitive.
    for point in [*source_points, *(section.get("key_points") or [])]:
        text = str(point.get("text") or "").strip()
        evidence_ids = list(dict.fromkeys(str(ref) for ref in point.get("evidence_ids", []) if ref))
        source_ids = list(dict.fromkeys(str(ref) for ref in point.get("source_ids", [])
                                        if ref and str(ref) not in evidence_ids))
        point["evidence_ids"] = evidence_ids
        point["source_ids"] = source_ids
        key = text.casefold().rstrip(" .;:")
        if not text:
            continue
        if key in seen:
            existing = seen[key]
            existing["evidence_ids"] = list(dict.fromkeys(
                [*existing.get("evidence_ids", []), *evidence_ids]
            ))
            existing["source_ids"] = list(dict.fromkeys(
                [*existing.get("source_ids", []), *source_ids]
            ))
            # Prefer the more informative grounded classification when a
            # model repeats a source claim under a generic gap label.
            rank = {"documented_fact": 0, "reported_observation": 1,
                    "hypothesis": 2, "historical_analogy": 3,
                    "proposal": 4, "verification_gap": 5}
            if rank.get(point.get("classification"), 5) < rank.get(existing.get("classification"), 5):
                existing["classification"] = point.get("classification")
            continue
        seen[key] = point
        combined.append(point)
    section["key_points"] = combined
    return section


def _task_spec(task_id: str, source_keys: tuple[str, ...]) -> AgentSpec:
    return AgentSpec(
        key=f"report_{task_id}",
        name=f"Report section: {_TASKS[task_id]['title']}",
        prompt_path=_PACKAGE / _TASKS[task_id]["prompt"],
        output_schema=ReportSectionDraft,
        upstream_context=source_keys,
    )


async def create_report(
    *,
    client: httpx.AsyncClient,
    settings: Settings,
    narrative: str,
    evidence: list[dict[str, Any]],
    prior_outputs: Mapping[str, dict[str, Any]],
    deterministic: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Plan and draft report sections as small, independently validated tasks."""
    if deterministic:
        # Bounded investigation: format existing outputs without five extra model tasks.
        sections = []
        for task_id in REPORT_TASK_ORDER:
            task_outputs = _task_source_outputs(task_id, prior_outputs)
            section_data = _fallback_section(task_id, task_outputs, evidence, narrative)
            section_data = _merge_source_detail(section_data, task_id, task_outputs, evidence, narrative)
            section = ReportSection.model_validate(section_data).model_dump(mode="json")
            section["title"] = _TASKS[task_id]["title"]
            sections.append(section)
        sections = _ground_sections(sections, {"agent_outputs": prior_outputs, "retrieved_evidence": evidence})
        output = ReportAgentOutput(executive_summary=sections[0]["summary"][:2200], sections=sections,
                                   human_approval_required=True).model_dump(mode="json")
        return output, [{"step": "report_assembly", "status": "deterministic_fallback"}]
    # The report schema already defines a fixed, validated four-part plan.
    # Avoid a separate planner call so report decomposition costs four model
    # calls, not five, while each section still gets its own prompt and budget.
    task_ids = list(REPORT_TASK_ORDER)
    trace: list[dict[str, Any]] = [{
        "step": "report_task_plan", "agent": "Report Task Planner",
        "status": "fixed_validated_plan",
    }]

    sections: list[dict[str, Any]] = []
    for task_id in task_ids:
        source_outputs = _task_source_outputs(task_id, prior_outputs)
        spec = _task_spec(task_id, tuple(source_outputs))
        try:
            draft = await invoke_agent(
                spec, client=client, settings=settings, narrative=narrative,
                evidence=evidence, prior_outputs=source_outputs,
            )
            # The task ID and heading come from the validated work plan, not from
            # free-form model text. This prevents sections from being mislabeled.
            draft["task_id"] = task_id
            draft["generation_method"] = "model"
            section = ReportSection.model_validate(draft).model_dump(mode="json")
            source_fallback = _fallback_section(task_id, source_outputs, evidence, narrative)
            if (not section.get("summary", "").strip()
                    or section["summary"].strip().casefold() in {
                        "preliminary summary assembled from validated source-agent outputs.",
                        "summary", "n/a", "not available",
                    }):
                section["summary"] = source_fallback["summary"]
            section = _merge_source_detail(section, task_id, source_outputs, evidence, narrative)
            trace.append({"step": f"report_{task_id}", "agent": spec.name, "status": "complete"})
        except Exception as exc:
            logger.warning(
                "Report section unavailable; using deterministic source summary; task=%s error_type=%s validation_details=%s",
                task_id, type(exc).__name__, getattr(exc, "diagnostic", "not_available"),
            )
            section_data = _fallback_section(task_id, source_outputs, evidence, narrative)
            section_data = _merge_source_detail(section_data, task_id, source_outputs, evidence, narrative)
            section = ReportSection.model_validate(section_data).model_dump(mode="json")
            trace.append({"step": f"report_{task_id}", "agent": spec.name, "status": "deterministic_fallback"})
        section["title"] = _TASKS[task_id]["title"]
        sections.append(section)

    sections = _ground_sections(sections, {"agent_outputs": prior_outputs, "retrieved_evidence": evidence})
    overview = next((item for item in sections if item["task_id"] == "overview"), None)
    executive_summary = (overview or {}).get("summary") or "Preliminary analysis assembled from the agent work products."
    output = ReportAgentOutput(
        executive_summary=executive_summary,
        sections=sections,
        human_approval_required=True,
    ).model_dump(mode="json")
    return output, trace
