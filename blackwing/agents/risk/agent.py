from pathlib import Path
from typing import Any

from blackwing.agents.base import AgentSpec
from .schema import RiskOutput


SPEC = AgentSpec(
    key="risk",
    name="Risk Agent",
    prompt_path=Path(__file__).with_name("prompt.md"),
    output_schema=RiskOutput,
    upstream_context=("investigation", "similar_case", "compliance"),
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
    return RiskOutput.model_validate(root).model_dump(mode="json")
