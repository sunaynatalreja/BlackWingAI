from pydantic import Field

from blackwing.contracts import APIModel


class CriticIssueOutput(APIModel):
    issue_type: str
    severity: str = "review"
    affected_output: str | None = None
    description: str
    suggested_correction: str | None = None


class CriticOutput(APIModel):
    issues: list[CriticIssueOutput] = Field(default_factory=list)
    missing_analysis: list[str] = Field(default_factory=list)
    overall_assessment: str = "requires_human_review"
