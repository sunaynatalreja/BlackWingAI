from pydantic import Field

from blackwing.contracts import APIModel


class ComplianceCheckOutput(APIModel):
    source_id: str | None = None
    source_title: str | None = None
    source_type: str = "unknown"
    jurisdiction: str | None = None
    effective_date: str | None = None
    section: str | None = None
    applicability_status: str = "requires_human_verification"
    rationale: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class ComplianceOutput(APIModel):
    jurisdiction: str | None = None
    overall_applicability: str = "not_assessed"
    checks: list[ComplianceCheckOutput] = Field(default_factory=list)
    missing_context: list[str] = Field(default_factory=list)
