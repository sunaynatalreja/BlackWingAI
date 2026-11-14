from pathlib import Path

from blackwing.agents.base import AgentSpec
from .schema import MitigationOutput


SPEC = AgentSpec(
    key="mitigation",
    name="Mitigation Agent",
    prompt_path=Path(__file__).with_name("prompt.md"),
    output_schema=MitigationOutput,
    upstream_context=("investigation", "risk"),
)
