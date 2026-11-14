from pathlib import Path

from blackwing.agents.base import AgentSpec
from blackwing.agents.similar_case.schema import SimilarCaseOutput

SPEC = AgentSpec(
    key="similar_case",
    name="Similar Case Analysis Agent",
    prompt_path=Path(__file__).with_name("prompt.md"),
    output_schema=SimilarCaseOutput,
    upstream_context=("investigation",),
)
