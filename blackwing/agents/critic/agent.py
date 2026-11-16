from pathlib import Path

from blackwing.agents.base import AgentSpec
from .schema import CriticOutput


SPEC = AgentSpec(
    key="critic",
    name="Critic Agent",
    prompt_path=Path(__file__).with_name("prompt.md"),
    output_schema=CriticOutput,
    upstream_context=("investigation", "similar_case", "compliance", "risk", "mitigation"),
)
