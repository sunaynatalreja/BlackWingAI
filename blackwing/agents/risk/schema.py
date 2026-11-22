from pydantic import Field

from blackwing.contracts import APIModel


class HazardOutput(APIModel):
    hazard_id: str
    description: str
    evidence_ids: list[str] = Field(default_factory=list)
    existing_controls: list[str] = Field(default_factory=list)
    control_gaps: list[str] = Field(default_factory=list)
    severity: int | None = Field(default=None, ge=1, le=5)
    severity_rationale: str = ""
    likelihood: int | None = Field(default=None, ge=1, le=5)
    likelihood_rationale: str = ""
    score: int | None = Field(default=None, ge=1, le=25)
    risk_rating: str | None = None


class RiskOutput(APIModel):
    hazards: list[HazardOutput] = Field(default_factory=list)
    risk_notes: str = ""
    assessment_status: str = "not_scored_missing_ratings"
    numeric_score: int | None = Field(default=None, ge=1, le=25)
    risk_rating: str | None = None
    scoring_method: str = "provisional_5x5_severity_x_likelihood_max_hazard"
    missing_inputs: list[str] = Field(default_factory=list)
