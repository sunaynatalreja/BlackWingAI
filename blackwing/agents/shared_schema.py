"""Small output primitives shared by agent-owned schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from blackwing.contracts import APIModel


class EvidenceState(StrEnum):
    DOCUMENTED_FACT = "documented_fact"
    REPORTED_OBSERVATION = "reported_observation"
    ANALYTICAL_INFERENCE = "analytical_inference"
    HYPOTHESIS = "hypothesis"
    UNKNOWN = "unknown"


class ClaimOutput(APIModel):
    text: str
    evidence_state: EvidenceState = EvidenceState.UNKNOWN
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    rationale: str | None = None


class TimelineEventOutput(APIModel):
    event: str
    time_reference: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_state: EvidenceState = EvidenceState.UNKNOWN
