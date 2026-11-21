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
    ReportPlanOutput,
    ReportSection,
    ReportSectionDraft,
)

logger = logging.getLogger("blackwing.agents.report")
_PACKAGE = Path(__file__).parent

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
            point["evidence_ids"] = [item for item in point.get("evidence_ids", []) if item in allowed_evidence]
            point["source_ids"] = [item for item in point.get("source_ids", []) if item in allowed_sources]
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


def _fallback_section(task_id: str, prior: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
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
    elif task_id == "findings_and_causes":
        summary = "The agent review does not establish a probable cause. Candidate explanations remain provisional until primary evidence is verified."
        points += _claim_points(investigation.get("hypotheses"), "hypothesis")
        points += _claim_points(investigation.get("contributing_factors"), "hypothesis")
        points += _claim_points(verification.get("claims_needing_source"), "verification_gap")
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
        score = risk.get("numeric_score")
        rating = risk.get("risk_rating")
        score_summary = (
            f"Provisional 5×5 risk score: {score}/25 ({rating}); highest scored hazard. "
            if score is not None else
            "Provisional risk score unavailable because severity or likelihood is missing. "
        )
        summary = score_summary + str(risk.get("risk_notes") or "Candidate hazards require human review.")
        points += _claim_points(risk.get("hazards"), "hypothesis")
        for recommendation in mitigation.get("recommendations", []):
            if isinstance(recommendation, dict) and recommendation.get("action"):
                points.append({
                    "text": str(recommendation["action"]), "classification": "proposal",
                    "evidence_ids": [], "source_ids": [],
                })
        points += _claim_points(verification.get("claims_needing_source"), "verification_gap")
        points += _claim_points(critic.get("issues"), "verification_gap")

    return {
        "task_id": task_id,
        "summary": str(summary),
        "key_points": points,
        "generation_method": "deterministic_fallback",
    }


def _merge_source_detail(section: dict[str, Any], task_id: str,
                         prior: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """Restore source-agent details a short-model draft may have omitted.

    These points are copied from already validated agent outputs; this step
    does not invent or upgrade claims. The report schema allows ten combined
    source and model points.
    """
    source_points = _fallback_section(task_id, prior).get("key_points", [])
    combined: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    # Source details lead so they remain visible if a model draft is repetitive.
    for point in [*source_points, *(section.get("key_points") or [])]:
        text = str(point.get("text") or "").strip()
        refs = tuple(sorted(str(ref) for ref in [
            *(point.get("evidence_ids") or []), *(point.get("source_ids") or [])
        ]))
        key = (text.casefold(), refs)
        if not text or key in seen:
            continue
        seen.add(key)
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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Plan and draft report sections as small, independently validated tasks."""
    planner = AgentSpec(
        key="report_planner",
        name="Report Task Planner",
        prompt_path=_PACKAGE / "planner.md",
        output_schema=ReportPlanOutput,
        upstream_context=(),
    )
    trace: list[dict[str, Any]] = []
    try:
        plan_data = await invoke_agent(
            planner, client=client, settings=settings, narrative=narrative,
            evidence=[], prior_outputs={},
        )
        task_ids = ReportPlanOutput.model_validate(plan_data).task_ids
        trace.append({"step": "report_task_plan", "agent": planner.name, "status": "complete"})
    except Exception as exc:
        logger.warning("Report task planner unavailable; using fixed task plan; error_type=%s", type(exc).__name__)
        task_ids = list(REPORT_TASK_ORDER)
        trace.append({"step": "report_task_plan", "agent": planner.name, "status": "deterministic_fallback"})

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
            section = _merge_source_detail(section, task_id, source_outputs)
            trace.append({"step": f"report_{task_id}", "agent": spec.name, "status": "complete"})
        except Exception as exc:
            logger.warning(
                "Report section unavailable; using deterministic source summary; task=%s error_type=%s",
                task_id, type(exc).__name__,
            )
            section = ReportSection.model_validate(_fallback_section(task_id, source_outputs)).model_dump(mode="json")
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
