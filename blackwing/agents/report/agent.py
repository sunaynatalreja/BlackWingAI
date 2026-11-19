from pathlib import Path

from blackwing.agents.base import AgentSpec
from .schema import ReportAgentOutput


SPEC = AgentSpec(
    key="report",
    name="Report Agent",
    prompt_path=Path(__file__).with_name("prompt.md"),
    output_schema=ReportAgentOutput,
    upstream_context=(
        "investigation", "similar_case", "compliance", "risk", "mitigation",
        "critic", "verification",
    ),
)
