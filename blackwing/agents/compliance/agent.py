from pathlib import Path
from typing import Any

from blackwing.agents.base import AgentSpec
from .schema import ComplianceCheckOutput, ComplianceOutput


SPEC = AgentSpec(
    key="compliance",
    name="Compliance Agent",
    prompt_path=Path(__file__).with_name("prompt.md"),
    output_schema=ComplianceOutput,
    upstream_context=("investigation",),
)


def normalize_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep schema fields from common Qwen Compliance variants, discard free-form extras.

    The grounding pass reattaches passage IDs from the selected regulatory source.
    Model-generated incident excerpts are deliberately discarded because they can
    confuse a historical analogy with evidence about the current incident.
    """
    root: Any = payload
    if not isinstance(root.get("checks"), list):
        for wrapper in ("compliance", "compliance_output", "result", "output"):
            candidate = root.get(wrapper)
            if isinstance(candidate, dict) and isinstance(candidate.get("checks"), list):
                root = candidate
                break

    allowed_check_fields = set(ComplianceCheckOutput.model_fields)
    checks = []
    for item in root.get("checks", []):
        if not isinstance(item, dict):
            continue
        # In particular, drop evidence_id, regulation_text and
        # incident_fact_evidence. Grounding resolves source_id to approved
        # source metadata and inserts the matching regulatory passage ID.
        checks.append({key: value for key, value in item.items() if key in allowed_check_fields})

    missing = root.get("missing_context", [])
    if isinstance(missing, str):
        missing = [missing]
    if not isinstance(missing, list):
        missing = []
    missing = [item for item in missing if isinstance(item, str)]
    explanation = root.get("explanation")
    if isinstance(explanation, str) and explanation.strip():
        missing.append(explanation.strip())

    return {
        "jurisdiction": root.get("jurisdiction"),
        "overall_applicability": root.get("overall_applicability", "not_assessed"),
        "checks": checks[:2],
        "missing_context": list(dict.fromkeys(missing))[:12],
    }


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
