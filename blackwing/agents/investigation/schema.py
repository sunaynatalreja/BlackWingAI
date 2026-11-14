from pydantic import Field

from blackwing.contracts import APIModel
from blackwing.agents.shared_schema import ClaimOutput, TimelineEventOutput


class InvestigationOutput(APIModel):
    incident_summary: str = ""
    timeline_events: list[TimelineEventOutput] = Field(default_factory=list)
    documented_facts: list[ClaimOutput] = Field(default_factory=list)
    reported_observations: list[ClaimOutput] = Field(default_factory=list)
    hypotheses: list[ClaimOutput] = Field(default_factory=list)
    contributing_factors: list[ClaimOutput] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
