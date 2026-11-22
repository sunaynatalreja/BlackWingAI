"""Small output primitives shared by agent-owned schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator

from blackwing.contracts import APIModel


class EvidenceState(StrEnum):
    DOCUMENTED_FACT = "documented_fact"
    REPORTED_OBSERVATION = "reported_observation"
    ANALYTICAL_INFERENCE = "analytical_inference"
    HYPOTHESIS = "hypothesis"
    UNKNOWN = "unknown"


def _canonical_evidence_state(value):
    # Normalize spelling format only. Unknown or stronger custom claims must
    # still fail validation; no unverified assertion is promoted to a fact.
    if isinstance(value, str):
        canonical = "_".join(value.strip().casefold().replace("-", " ").replace("_", " ").split())
        if canonical in {state.value for state in EvidenceState}:
            return canonical
    return value


class ClaimOutput(APIModel):
    text: str
    evidence_state: EvidenceState = EvidenceState.UNKNOWN
    _normalize_state = field_validator("evidence_state", mode="before")(_canonical_evidence_state)
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    rationale: str | None = None


class TimelineEventOutput(APIModel):
    event: str
    time_reference: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_state: EvidenceState = EvidenceState.UNKNOWN
    _normalize_state = field_validator("evidence_state", mode="before")(_canonical_evidence_state)
