from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from blackwing.contracts import APIModel


def normalize_report_section(payload: dict) -> dict:
    """Accept shape aliases; provenance is checked separately during assembly."""
    summary = next((payload.get(k) for k in (
        'summary', 'section_summary', 'summary_sentence', 'executive_summary',
        'findings_summary', 'conclusion', 'key_message', 'overview', 'narrative',
    ) if isinstance(payload.get(k), str) and payload[k].strip()), '')
    raw = payload.get('key_points', payload.get('findings', payload.get('observations', [])))
    raw = [raw] if isinstance(raw, (str, dict)) else raw
    points = []
    aliases = {'fact': 'documented_fact', 'verified_fact': 'documented_fact',
               'reported': 'reported_observation', 'observation': 'reported_observation',
               'inference': 'hypothesis', 'analysis': 'hypothesis',
               'analogy': 'historical_analogy', 'recommendation': 'proposal',
               'unknown': 'verification_gap', 'gap': 'verification_gap'}
    allowed = {'documented_fact', 'reported_observation', 'hypothesis',
               'historical_analogy', 'proposal', 'verification_gap'}
    def ids(value):
        value = [value] if isinstance(value, str) else value
        return list(dict.fromkeys(x.strip() for x in value if isinstance(x, str) and x.strip())) if isinstance(value, list) else []
    for item in raw if isinstance(raw, list) else []:
        item = {'text': item} if isinstance(item, str) else item
        if not isinstance(item, dict):
            continue
        text = next((item[k] for k in ('text', 'claim', 'description', 'finding')
                     if isinstance(item.get(k), str) and item[k].strip()), '')
        if not text:
            continue
        label = str(item.get('classification') or item.get('type') or 'verification_gap').casefold()
        label = aliases.get(label, label)
        points.append({'text': text.strip(), 'classification': label if label in allowed else 'verification_gap',
                       'evidence_ids': ids(item.get('evidence_ids') or item.get('citations')),
                       'source_ids': ids(item.get('source_ids'))})
    if not summary and points:
        # The section contract requires a summary. Reuse the first already
        # returned point rather than failing the whole 8B report task or adding
        # new content.
        summary = points[0]['text']
    if not summary:
        raise ValueError('Report section requires a nonempty summary')
    # Section ID/title are assigned by code, never normalized from model text.
    return {
        'summary': summary.strip(),
        'key_points': points,
    }


ReportTaskId = Literal[
    "overview",
    "findings_and_causes",
    "comparisons_and_compliance",
    "risk_and_follow_up",
]

REPORT_TASK_ORDER: tuple[str, ...] = (
    "overview",
    "findings_and_causes",
    "comparisons_and_compliance",
    "risk_and_follow_up",
)


def normalize_report_plan_payload(payload: dict) -> tuple[dict, bool]:
    """Repair the common ``task_order`` alias and enforce canonical order."""
    if not isinstance(payload, dict):
        return payload, False
    normalized = dict(payload)
    used_alias = False
    if "task_ids" not in normalized and isinstance(normalized.get("task_order"), list):
        normalized["task_ids"] = normalized.pop("task_order")
        used_alias = True
    tasks = normalized.get("task_ids")
    if isinstance(tasks, list) and len(tasks) == len(REPORT_TASK_ORDER) and set(tasks) == set(REPORT_TASK_ORDER):
        if tasks != list(REPORT_TASK_ORDER):
            normalized["task_ids"] = list(REPORT_TASK_ORDER)
            used_alias = True
    return normalized, used_alias


class ReportPlanOutput(APIModel):
    """A constrained plan: task IDs are fixed and section work is independent."""

    task_ids: list[ReportTaskId] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def require_each_task_once(self) -> "ReportPlanOutput":
        if set(self.task_ids) != set(REPORT_TASK_ORDER):
            raise ValueError("The plan must include each report task exactly once")
        return self


class ReportPoint(APIModel):
    text: str
    classification: Literal[
        "reported_observation",
        "documented_fact",
        "hypothesis",
        "historical_analogy",
        "proposal",
        "verification_gap",
    ] = "verification_gap"
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class ReportSectionDraft(APIModel):
    summary: str = Field(min_length=1)
    key_points: list[ReportPoint] = Field(default_factory=list)


class ReportSection(ReportSectionDraft):
    task_id: ReportTaskId
    title: str = ""
    generation_method: Literal["model", "deterministic_fallback"] = "model"


class ReportAgentOutput(APIModel):
    executive_summary: str
    sections: list[ReportSection] = Field(min_length=4, max_length=4)
    human_approval_required: Literal[True] = True
    report_notice: str = (
        "Preliminary AI-assisted analysis. Verify sources and obtain qualified human review."
    )
