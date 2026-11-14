from pydantic import Field
from typing import Literal

from blackwing.contracts import APIModel


class SimilarCaseMatch(APIModel):
    source_id: str
    title: str
    source_url: str | None = None
    similarity: str
    differences_or_unknowns: list[str] = Field(default_factory=list)
    source_findings: list[str] = Field(default_factory=list)


class SimilarCaseOutput(APIModel):
    case_status: Literal[
        "unseen_case_with_similar_historical_cases",
        "unseen_case_no_historical_match",
        "existing_case_match",
    ] = "unseen_case_no_historical_match"
    matches: list[SimilarCaseMatch] = Field(default_factory=list)
    assessment: str = ""
    factors_to_verify: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
