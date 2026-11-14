from pydantic import Field

from blackwing.contracts import APIModel


class MitigationOptionOutput(APIModel):
    hazard_ids: list[str] = Field(default_factory=list)
    contributing_factor: str | None = None
    action: str
    classification: str = "preventive"
    responsible_area: str | None = None
    expected_effect: str = ""
    dependency: str | None = None
    priority: str = "proposed"
    effectiveness_indicator: str | None = None


class MitigationOutput(APIModel):
    recommendations: list[MitigationOptionOutput] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
