from pathlib import Path
from typing import Any

from blackwing.agents.base import AgentSpec
from .schema import CriticOutput


SPEC = AgentSpec(
    key="critic",
    name="Critic Agent",
    prompt_path=Path(__file__).with_name("prompt.md"),
    output_schema=CriticOutput,
    upstream_context=("investigation", "similar_case", "compliance", "risk", "mitigation"),
)


def normalize_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize common small-model Critic JSON variations before schema validation."""
    root: Any = payload
    if not isinstance(root.get("issues"), list):
        for wrapper in ("critic", "critic_output", "result", "output"):
            candidate = root.get(wrapper)
            if isinstance(candidate, dict) and isinstance(candidate.get("issues"), list):
                root = candidate
                break

    normalized_issues = []
    for item in root.get("issues", []):
        if not isinstance(item, dict):
            continue
        issue_type = item.get("issue_type", item.get("type", "model_reported_issue"))
        description = item.get("description", item.get("details", item.get("problem", item.get("finding"))))
        if not isinstance(issue_type, str) or not issue_type.strip():
            issue_type = "model_reported_issue"
        if not isinstance(description, str) or not description.strip():
            # Drop empty issue records. Deterministic audit findings are added
            # after validation and do not depend on this model-generated list.
            continue
        severity = item.get("severity", "review")
        affected = item.get("affected_output", item.get("affected_agent"))
        suggested = item.get("suggested_correction", item.get("suggested_fix", item.get("correction")))
        normalized_issues.append({
            "issue_type": issue_type.strip(),
            "severity": severity if isinstance(severity, str) else "review",
            "affected_output": affected if isinstance(affected, str) else None,
            "description": description.strip(),
            "suggested_correction": suggested if isinstance(suggested, str) else None,
        })

    missing = root.get("missing_analysis", [])
    if isinstance(missing, str):
        missing = [missing]
    if not isinstance(missing, list):
        missing = []
    missing = [item for item in missing if isinstance(item, str)][:5]
    assessment = root.get("overall_assessment", "requires_human_review")
    if not isinstance(assessment, str) or not assessment.strip():
        assessment = "requires_human_review"
    return {
        "issues": normalized_issues[:4],
        "missing_analysis": missing,
        "overall_assessment": assessment,
    }
