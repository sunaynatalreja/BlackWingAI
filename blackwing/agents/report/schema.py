from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from blackwing.contracts import APIModel


_POINT_CLASSES = {
    "reported_observation", "documented_fact", "hypothesis",
    "historical_analogy", "proposal", "verification_gap",
}


def normalize_report_section(payload: dict) -> dict:
    """Normalize common small-model JSON variants before strict validation.

    Claim text remains model-authored and citations are still filtered against
    the actual upstream evidence packet by the report workflow.
    """
    summary = next((payload.get(key) for key in (
        "summary", "section_summary", "summary_sentence", "executive_summary",
        "findings_summary", "conclusion", "key_message", "overview", "narrative",
    ) if isinstance(payload.get(key), str) and payload.get(key).strip()), "")
    raw_points = payload.get("key_points", payload.get("findings", payload.get("observations", [])))
    if isinstance(raw_points, (str, dict)):
        raw_points = [raw_points]
    points = []
    if isinstance(raw_points, list):
        for item in raw_points:
            if isinstance(item, str):
                item = {"text": item}
            if not isinstance(item, dict):
                continue
            text = item.get("text") or item.get("claim") or item.get("description") or item.get("finding")
            if not isinstance(text, str) or not text.strip():
                continue
            classification = str(item.get("classification") or item.get("type") or "verification_gap").casefold()
            classification = {
                "fact": "documented_fact", "verified_fact": "documented_fact",
                "reported": "reported_observation", "observation": "reported_observation",
                "inference": "hypothesis", "analysis": "hypothesis",
                "analogy": "historical_analogy", "recommendation": "proposal",
                "unknown": "verification_gap", "gap": "verification_gap",
            }.get(classification, classification)
            if classification not in _POINT_CLASSES:
                classification = "verification_gap"
            evidence_ids = item.get("evidence_ids") or item.get("citations") or []
            source_ids = item.get("source_ids") or []
            if isinstance(evidence_ids, str):
                evidence_ids = [evidence_ids]
            if isinstance(source_ids, str):
                source_ids = [source_ids]
            points.append({"text": text.strip(), "classification": classification,
                           "evidence_ids": [str(x) for x in evidence_ids if isinstance(x, (str, int))],
                           "source_ids": [str(x) for x in source_ids if isinstance(x, (str, int))]})
    return {"summary": summary.strip(), "key_points": points}

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
    """Model-authored section body; code assigns the task ID and title."""
    summary: str
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
