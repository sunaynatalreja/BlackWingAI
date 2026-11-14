from pathlib import Path

from blackwing.agents.base import AgentSpec
from .schema import RiskOutput


SPEC = AgentSpec(
    key="risk",
    name="Risk Agent",
    prompt_path=Path(__file__).with_name("prompt.md"),
    output_schema=RiskOutput,
    upstream_context=("investigation", "similar_case", "compliance"),
)
