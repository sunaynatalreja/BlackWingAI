"""Investigation Agent registration and backward-compatible citation guard."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from blackwing.agents.base import AgentSpec
from blackwing.agents.investigation.grounding import ground_investigation_output
from blackwing.agents.evidence_utils import citation_map, is_direct, source_id
from blackwing.agents.investigation.schema import InvestigationOutput

SPEC = AgentSpec(
    key="investigation", name="Investigation Agent",
    prompt_path=Path(__file__).with_name("prompt.md"),
    output_schema=InvestigationOutput, upstream_context=(),
)


def remove_historical_case_evidence_ids(output: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Legacy public name retained for Sunayana's original contract tests.

    Removed historical sources are kept as *analogy references*, never as
    current-case evidence_ids. Production orchestration calls the stricter
    ground_investigation_output() instead.
    """
    sanitized = ground_investigation_output(output, evidence)
    aliases = citation_map(evidence)
    for field in ("documented_facts", "reported_observations", "hypotheses", "contributing_factors"):
        for before, after in zip(output.get(field, []), sanitized.get(field, [])):
            rejected_historical = []
            for ref in before.get("evidence_ids", []):
                match = aliases.get(str(ref))
                if match and not is_direct(match[1]):
                    rejected_historical.append(source_id(match[1]))
            if rejected_historical:
                after["source_ids"] = list(dict.fromkeys([*after.get("source_ids", []), *rejected_historical]))
                note = "Historical analogy only; this source describes a different incident and does not establish this factor in the current case."
                after["rationale"] = (str(after.get("rationale") or "").strip() + " " + note).strip()
    return sanitized
