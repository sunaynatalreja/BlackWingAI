"""Public request and response models for the versioned BlackWing API."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from blackwing.contracts import APIModel
from blackwing.agents.shared_schema import ClaimOutput, EvidenceState, TimelineEventOutput
from blackwing.agents.investigation.schema import InvestigationOutput
from blackwing.agents.compliance.schema import ComplianceCheckOutput, ComplianceOutput
from blackwing.agents.risk.schema import HazardOutput, RiskOutput
from blackwing.agents.mitigation.schema import MitigationOptionOutput, MitigationOutput
from blackwing.agents.critic.schema import CriticIssueOutput, CriticOutput
from blackwing.agents.verification.schema import VerificationOutput
from blackwing.agents.report.schema import ReportAgentOutput
from blackwing.agents.similar_case.schema import SimilarCaseOutput


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class IncidentCategory(StrEnum):
    RUNWAY_INCURSION = "runway_incursion"


class CaseStatus(StrEnum):
    DRAFT = "draft"
    ANALYSIS_RUNNING = "analysis_running"
    ANALYSIS_COMPLETE = "analysis_complete"
    REVIEW_PENDING = "review_pending"
    REVIEWED = "reviewed"
    ANALYSIS_FAILED = "analysis_failed"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ReviewDecision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"


class EvidenceInput(APIModel):
    title: str = Field(min_length=1, max_length=300)
    document_type: str = Field(min_length=1, max_length=80)
    authority_type: str | None = Field(default=None, max_length=80)
    source_status: str | None = Field(default=None, max_length=80)
    version: str | None = Field(default=None, max_length=100)
    source_url: str | None = Field(default=None, max_length=2048)
    source_id: str | None = Field(default=None, max_length=200)
    jurisdiction: str | None = Field(default=None, max_length=100)
    document_date: date | None = None
    effective_date: date | None = None
    page_number: int | None = Field(default=None, ge=1)
    section_reference: str | None = Field(default=None, max_length=200)
    excerpt: str = Field(min_length=1, max_length=20_000)


class CaseCreate(APIModel):
    source_incident_id: str | None = Field(default=None, min_length=3, max_length=100, description="External incident source identifier for matching indexed factual reports.")
    category: IncidentCategory = IncidentCategory.RUNWAY_INCURSION
    narrative: str = Field(min_length=20, max_length=30_000)
    incident_date: date | None = None
    jurisdiction: str | None = Field(default=None, max_length=100)
    location: str | None = Field(default=None, max_length=300)
    evidence: list[EvidenceInput] = Field(default_factory=list, max_length=30)


class CasePatch(APIModel):
    source_incident_id: str | None = Field(default=None, min_length=3, max_length=100)
    narrative: str | None = Field(default=None, min_length=20, max_length=30_000)
    incident_date: date | None = None
    jurisdiction: str | None = Field(default=None, max_length=100)
    location: str | None = Field(default=None, max_length=300)


class EvidenceResponse(EvidenceInput):
    evidence_id: str
    created_at: datetime


class KnowledgeDocumentCreate(EvidenceInput):
    content: str = Field(min_length=20, max_length=500_000)


class KnowledgeDocumentResponse(APIModel):
    document_id: str
    title: str
    document_type: str
    authority_type: str | None = None
    source_status: str | None = None
    version: str | None = None
    source_url: str | None = None
    source_id: str | None = None
    jurisdiction: str | None = None
    document_date: date | None = None
    effective_date: date | None = None
    page_number: int | None = None
    section_reference: str | None = None
    content_length: int
    created_at: datetime


class KnowledgeSearchRequest(APIModel):
    query: str = Field(min_length=3, max_length=2000)
    document_types: list[str] = Field(default_factory=list, max_length=10)
    jurisdiction: str | None = Field(default=None, max_length=100)
    incident_date: date | None = None
    limit: int = Field(default=5, ge=1, le=20)


class KnowledgeSearchResponse(APIModel):
    results: list[dict[str, Any]]
    retrieval_method: str
    limitations: list[str]


class CaseResponse(APIModel):
    case_id: str
    source_incident_id: str | None = None
    category: IncidentCategory
    status: CaseStatus
    narrative: str
    incident_date: date | None = None
    jurisdiction: str | None = None
    location: str | None = None
    evidence: list[EvidenceResponse]
    created_at: datetime
    updated_at: datetime


class CaseListResponse(APIModel):
    items: list[CaseResponse]
    total: int
    limit: int
    offset: int


class RunCreate(APIModel):
    workflow: str = Field(default="runway_incursion_investigation", pattern="^runway_incursion_investigation$")
    include_retrieval: bool = True


class RunResponse(APIModel):
    run_id: str
    case_id: str
    status: RunStatus
    workflow: str
    model: str | None = None
    progress: list[dict[str, Any]]
    error_code: str | None = None
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    links: dict[str, str]


class TraceEvent(APIModel):
    sequence: int
    step: str
    agent: str | None = None
    status: str
    model: str | None = None
    detail: str | None = None
    timestamp: datetime


class TraceResponse(APIModel):
    run_id: str
    events: list[TraceEvent]


class FindingOverride(APIModel):
    path: str = Field(min_length=1, max_length=300)
    action: str = Field(pattern="^(accept|edit|reject)$")
    replacement_value: Any | None = None
    rationale: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def override_requires_rationale(self) -> "FindingOverride":
        if self.action in {"edit", "reject"} and not (self.rationale or "").strip():
            raise ValueError("A rationale is required when editing or rejecting a finding")
        return self


class ReviewCreate(APIModel):
    reviewer_id: str = Field(min_length=1, max_length=200)
    decision: ReviewDecision
    rationale: str = Field(min_length=1, max_length=4000)
    overrides: list[FindingOverride] = Field(default_factory=list, max_length=100)


class ReviewResponse(ReviewCreate):
    review_id: str
    case_id: str
    created_at: datetime


class AgentOutputs(APIModel):
    investigation: InvestigationOutput
    similar_case: SimilarCaseOutput
    compliance: ComplianceOutput
    risk: RiskOutput
    mitigation: MitigationOutput
    critic: CriticOutput
    verification: VerificationOutput


class AnalysisTraceEntry(APIModel):
    step: str
    agent: str | None = None
    model: str | None = None
    status: str
    retrieved_document_count: int | None = None


class AnalysisReport(APIModel):
    case_id: str
    status: str
    workflow: str
    model: str
    agent_outputs: AgentOutputs
    generated_report: ReportAgentOutput | None = None
    trace: list[AnalysisTraceEntry]
    limitations: list[str]


class ReportResponse(APIModel):
    case_id: str
    run_id: str
    case_status: CaseStatus
    analysis_status: RunStatus
    report: AnalysisReport
    reviews: list[ReviewResponse]
    is_final: bool = False
    notice: str


class ErrorResponse(APIModel):
    error: str
    message: str
    request_id: str
