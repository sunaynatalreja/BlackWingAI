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
    # Normalize known spelling variants. A label outside the closed enum is
    # conservatively treated as unknown rather than failing the whole agent
    # output or being promoted to a stronger evidence state.
    if isinstance(value, str):
        canonical = "_".join(value.strip().casefold().replace("-", " ").replace("_", " ").split())
        if canonical in {state.value for state in EvidenceState}:
            return canonical
    return EvidenceState.UNKNOWN.value


def normalize_evidence_state_labels(payload):
    """Conservatively normalize invalid nested evidence-state labels in-place.

    Returns the number of values changed so the runtime can record that the
    model used a non-contract label. No fact text or citations are modified.
    """
    changes = 0

    def visit(value):
        nonlocal changes
        if isinstance(value, dict):
            if "evidence_state" in value:
                original = value["evidence_state"]
                normalized = _canonical_evidence_state(original)
                if original != normalized:
                    value["evidence_state"] = normalized
                    changes += 1
            for key, child in value.items():
                if key != "evidence_state":
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return changes


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
