from pathlib import Path
from typing import Any

from blackwing.agents.base import AgentSpec
from blackwing.agents.risk.scoring import LIKELIHOOD_LEVELS, SEVERITY_LEVELS
from .schema import RiskOutput


SPEC = AgentSpec(
    key="risk",
    name="Risk Agent",
    prompt_path=Path(__file__).with_name("prompt.md"),
    output_schema=RiskOutput,
    upstream_context=("investigation", "similar_case", "compliance"),
)


def _rating_level(value: Any, levels: dict[int, str]) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 5:
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    for number, label in levels.items():
        if normalized == label.casefold() or normalized.startswith(label.casefold() + " "):
            return number
    if normalized and normalized[0] in "12345" and (len(normalized) == 1 or normalized[1] in " -("):
        return int(normalized[0])
    if normalized.isdigit() and 1 <= int(normalized) <= 5:
        return int(normalized)
    return None


def normalize_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate legacy Risk response fields into the current strict schema."""
    root: Any = payload
    if not isinstance(root.get("hazards"), list):
        for wrapper in ("risk", "risk_output", "result", "output"):
            candidate = root.get(wrapper)
            if isinstance(candidate, dict) and isinstance(candidate.get("hazards"), list):
                root = candidate
                break

    hazards = []
    missing_inputs = root.get("missing_inputs", [])
    if isinstance(missing_inputs, str):
        missing_inputs = [missing_inputs]
    if not isinstance(missing_inputs, list):
        missing_inputs = []
    missing_inputs = [item for item in missing_inputs if isinstance(item, str)]
    notes = [item for item in (root.get("risk_notes"), root.get("explanation"))
             if isinstance(item, str) and item.strip()]

    for index, item in enumerate(root.get("hazards", [])[:3], 1):
        if not isinstance(item, dict):
            continue
        description = item.get("description") or item.get("hazard")
        if not isinstance(description, str) or not description.strip():
            continue
        description = description.strip()
        consequences = item.get("consequences")
        if isinstance(consequences, str) and consequences.strip():
            description += f" Potential consequence: {consequences.strip()}"
        if not description.casefold().startswith(("potential", "hypothesis", "unverified")):
            description = "Potential hazard hypothesis: " + description

        evidence_ids = item.get("evidence_ids", [])
        if isinstance(evidence_ids, str):
            evidence_ids = [evidence_ids]
        if not isinstance(evidence_ids, list):
            evidence_ids = []
        evidence_ids = [ref for ref in evidence_ids if isinstance(ref, str)]

        controls = item.get("existing_controls", [])
        if isinstance(controls, str):
            controls = [controls]
        if not isinstance(controls, list):
            controls = []
        controls = [control for control in controls if isinstance(control, str)]
        if item.get("existing_barriers"):
            # Legacy output conflated general guidance with controls actually
            # implemented at the airport; don't promote it to an existing control.
            missing_inputs.append("Verify whether cited general guidance was implemented as an operational control.")

        gaps = item.get("control_gaps", [])
        if isinstance(gaps, str):
            gaps = [gaps]
        if not isinstance(gaps, list):
            gaps = []
        gaps = [gap for gap in gaps if isinstance(gap, str)]
        hazards.append({
            "hazard_id": str(item.get("hazard_id") or f"H-{index:02d}"),
            "description": description,
            "evidence_ids": evidence_ids,
            "existing_controls": controls,
            "control_gaps": gaps,
            "severity": _rating_level(item.get("severity"), SEVERITY_LEVELS),
            "severity_rationale": str(item.get("severity_rationale") or ""),
            "likelihood": _rating_level(item.get("likelihood"), LIKELIHOOD_LEVELS),
            "likelihood_rationale": str(item.get("likelihood_rationale") or ""),
        })

    if hazards:
        notes.append("Candidate hazards are hypotheses, not established causes or observed control failures.")
    return RiskOutput(
        hazards=hazards,
        risk_notes=" ".join(dict.fromkeys(notes)),
        assessment_status="provisional",
        numeric_score=None,
        risk_rating=None,
        scoring_method="provisional_5x5_severity_x_likelihood_max_hazard",
        missing_inputs=list(dict.fromkeys(missing_inputs))[:12],
    ).model_dump(mode="json")
