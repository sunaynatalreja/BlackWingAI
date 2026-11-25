from typing import Literal

from pydantic import Field

from blackwing.contracts import APIModel


class HazardOutput(APIModel):
    hazard_id: str
    description: str
    evidence_ids: list[str] = Field(default_factory=list)
    existing_controls: list[str] = Field(default_factory=list)
    control_gaps: list[str] = Field(default_factory=list)
    severity: int | None = Field(default=None, ge=1, le=5)
    likelihood: Literal["A", "B", "C", "D", "E"] | None = None
    severity_basis: str = ""
    likelihood_basis: str = ""
    risk_index: str | None = None
    risk_level: str | None = None
    scoring_status: str = "unassessed"


class RiskOutput(APIModel):
    hazards: list[HazardOutput] = Field(default_factory=list)
    risk_notes: str = ""
    assessment_status: str = "provisional"
    numeric_score: None = None
    missing_inputs: list[str] = Field(default_factory=list)
