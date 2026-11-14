from pathlib import Path

from blackwing.agents.base import AgentSpec
from .schema import ComplianceOutput


SPEC = AgentSpec(
    key="compliance",
    name="Compliance Agent",
    prompt_path=Path(__file__).with_name("prompt.md"),
    output_schema=ComplianceOutput,
    upstream_context=("investigation",),
)


def unassessed_output() -> dict[str, object]:
    """Return a safe result when no regulatory or procedural source is available."""
    return ComplianceOutput(
        overall_applicability="not_assessed",
        checks=[],
        missing_context=[
            "No regulatory, procedural, or guidance source text was supplied or retrieved.",
            "Confirm the governing jurisdiction and the source versions effective on the incident date.",
        ],
    ).model_dump(mode="json")
