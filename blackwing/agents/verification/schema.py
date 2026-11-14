from pydantic import Field

from blackwing.contracts import APIModel


class VerificationOutput(APIModel):
    claims_needing_source: list[str] = Field(default_factory=list)
    unsupported_conclusions: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    completeness_status: str = "incomplete"
    human_approval_required: bool = True
