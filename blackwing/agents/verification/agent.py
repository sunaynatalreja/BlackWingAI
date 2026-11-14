from pathlib import Path

from blackwing.agents.base import AgentSpec
from .schema import VerificationOutput


SPEC = AgentSpec(
    key="verification",
    name="Verification Agent",
    prompt_path=Path(__file__).with_name("prompt.md"),
    output_schema=VerificationOutput,
    upstream_context=("investigation", "compliance", "risk", "mitigation", "critic"),
)
