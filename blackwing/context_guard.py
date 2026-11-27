"""Deterministic scope and harmful-intent guard for BlackWing inputs."""
from __future__ import annotations

import re
from collections.abc import Iterable


REJECTION = (
    "Not a relevant context for BlackWing AI. Submit an aviation safety incident, "
    "operational event, or a question about aviation safety evidence."
)

_AVIATION = re.compile(
    r"\b(?:aviation|aircraft|airplane|aeroplane|airliner|airline|flight|airport|runway|"
    r"taxiway|apron|airfield|pilot|flightcrew|crew|cockpit|air traffic control|atc|"
    r"controller|tower|takeoff|take-off|landing|go around|go-around|airspace|"
    r"helicopter|rotorcraft|engine|airspeed|altitude|approach|departure|arrival)\b",
    re.I,
)
_INCIDENT = re.compile(
    r"\b(?:incident|accident|occurrence|event|investigat\w*|report(?:ed|ing)?|"
    r"safety|hazard|near miss|collision|incursion|runway conflict|emergency|"
    r"deviation|clearance|overrun|undershoot|rejected takeoff|bird strike|"
    r"loss of control|engine failure|smoke|fire|evacuat\w*|injur\w*|damage|"
    r"crossed|crossing|entered|stopped|departed|landed|aborted|rejected|"
    r"observed|reported|occurred|failed|failure|separation)\b",
    re.I,
)
_HARMFUL_REQUEST = re.compile(
    r"(?:\b(?:ways?|methods?|steps?|instructions?|how\s+to|help\s+me|tell\s+me\s+how)"
    r"\b.{0,100}\b(?:commit\s+suicide|suicid\w*|kill\s+myself|self[- ]harm|"
    r"hurt\s+myself|build\s+(?:a\s+)?bomb|make\s+(?:a\s+)?bomb|plan\s+(?:a\s+)?"
    r"terrorist\s+attack|plan\s+(?:an?\s+)?attack|carry\s+out\s+(?:a\s+)?attack)\b"
    r"|\b(?:commit\s+suicide|kill\s+myself|self[- ]harm|hurt\s+myself)\b)",
    re.I | re.S,
)


class IrrelevantContext(ValueError):
    """Input is outside BlackWing's aviation-safety scope or requests harm."""


def rejection_reason(
    texts: str | Iterable[str], *, request_text: str | None = None,
) -> str | None:
    """Return the user-facing rejection for harmful/off-topic input, else None.

    Sensitive terms alone do not cause rejection: an aviation incident report may
    legitimately discuss terrorism, suicide, injury, or other serious events.
    The rejection targets harmful instructions and inputs lacking an aviation
    event/safety context.
    """
    if isinstance(texts, str):
        text = texts
    else:
        text = "\n".join(str(value or "") for value in texts)
    text = text.strip()
    if not text:
        return REJECTION
    # Check intent only in the user's request, not quoted source documents.
    if _HARMFUL_REQUEST.search(request_text if request_text is not None else text):
        return REJECTION
    if not _AVIATION.search(text) or not _INCIDENT.search(text):
        return REJECTION
    return None


def require_relevant_context(
    texts: str | Iterable[str], *, request_text: str | None = None,
) -> None:
    reason = rejection_reason(texts, request_text=request_text)
    if reason:
        raise IrrelevantContext(reason)
