"""Link every proposed mitigation to an actual risk hazard or documented factor."""
from __future__ import annotations

from typing import Any, Mapping
from blackwing.agents.mitigation.schema import MitigationOutput


def ground_mitigation_output(output: Mapping[str, Any], prior_outputs: Mapping[str, Any]) -> dict[str, Any]:
    result = MitigationOutput.model_validate(output).model_dump(mode="json")
    risk = prior_outputs.get("risk") or {}
    investigation = prior_outputs.get("investigation") or {}
    hazards = {str(h.get("hazard_id")): h for h in risk.get("hazards", []) if h.get("hazard_id")}
    factors = [str(c.get("text", "")).strip() for c in investigation.get("contributing_factors", [])
               if c.get("evidence_ids") and c.get("text")]
    assumptions = list(result.get("assumptions") or [])
    valid = []
    seen = set()
    for rec in result["recommendations"]:
        linked = list(dict.fromkeys(x for x in rec["hazard_ids"] if x in hazards))
        factor = str(rec.get("contributing_factor") or "").strip()
        if factor and factor not in factors:
            factor = ""
        # Do not invent hazard IDs and never present unsupported free-standing
        # recommendations as operational actions.
        if not linked and not factor:
            assumptions.append("Omitted recommendation without a traceable hazard or evidenced contributing factor: "
                               + str(rec.get("action") or "")[:100])
            continue
        rec["hazard_ids"] = linked
        rec["contributing_factor"] = factor or None
        rec["classification"] = (rec.get("classification") if rec.get("classification") in {"preventive", "corrective"}
                                 else "proposed")
        rec["responsible_area"] = None  # No authority/owner assignment from LLM.
        if not any((hazards[x].get("evidence_ids") or []) for x in linked) and not factor:
            rec["priority"] = "proposed_requires_evidence"
            assumptions.append("Risk evidence missing for proposal linked to " + ", ".join(linked))
        else:
            # Qualitative priority is still tentative without a validated risk matrix.
            rec["priority"] = "proposed_for_human_review"
        key = (tuple(linked), factor, str(rec["action"]).strip().lower())
        if key not in seen:
            seen.add(key)
            valid.append(rec)
    result["recommendations"] = valid[:5]
    result["assumptions"] = list(dict.fromkeys(assumptions))[:12]
    return MitigationOutput.model_validate(result).model_dump(mode="json")
