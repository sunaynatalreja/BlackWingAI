from pydantic import Field

from blackwing.contracts import APIModel


class HazardOutput(APIModel):
    hazard_id: str
    description: str
    evidence_ids: list[str] = Field(default_factory=list)
    existing_controls: list[str] = Field(default_factory=list)
    control_gaps: list[str] = Field(default_factory=list)


class RiskOutput(APIModel):
    hazards: list[HazardOutput] = Field(default_factory=list)
    risk_notes: str = ""
    assessment_status: str = "provisional"
    numeric_score: None = None
    missing_inputs: list[str] = Field(default_factory=list)
