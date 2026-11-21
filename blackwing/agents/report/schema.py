from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from blackwing.contracts import APIModel

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
    task_id: ReportTaskId
    summary: str
    key_points: list[ReportPoint] = Field(default_factory=list)


class ReportSection(ReportSectionDraft):
    title: str = ""
    generation_method: Literal["model", "deterministic_fallback"] = "model"


class ReportAgentOutput(APIModel):
    executive_summary: str
    sections: list[ReportSection] = Field(min_length=4, max_length=4)
    human_approval_required: Literal[True] = True
    report_notice: str = (
        "Preliminary AI-assisted analysis. Verify sources and obtain qualified human review."
    )
