from pathlib import Path
from typing import Any

from blackwing.agents.base import AgentSpec
from .schema import RiskOutput


SPEC = AgentSpec(
    key="risk",
    name="Risk Agent",
    prompt_path=Path(__file__).with_name("prompt.md"),
    output_schema=RiskOutput,
    # Risk ratings use current-case facts only, never historical comparisons.
    upstream_context=("investigation",),
)


def normalize_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate categorical inputs; never translate numeric product ratings."""
    root = payload
    if "hazards" not in root:
        for wrapper in ("risk", "risk_output", "result", "output"):
            candidate = root.get(wrapper)
            if isinstance(candidate, dict) and "hazards" in candidate:
                root = candidate
                break
    # Small instruct models sometimes emit a scalar for list-valued fields.
    # Normalize only unambiguous single strings; citation identity is still
    # checked against the supplied evidence packet downstream.
    root = dict(root)
    for hazard in root.get("hazards") or []:
        if not isinstance(hazard, dict):
            continue
        for key in ("evidence_ids", "existing_controls", "control_gaps"):
            if isinstance(hazard.get(key), str):
                value = hazard[key].strip()
                hazard[key] = [value] if value else []
        for key in ("evidence_ids", "existing_controls", "control_gaps"):
            if hazard.get(key) is None:
                hazard[key] = []
    return RiskOutput.model_validate(root).model_dump(mode="json")
