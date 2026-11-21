"""Provisional, transparent 5x5 risk score calculations."""

from __future__ import annotations

from typing import Any


SEVERITY_LEVELS = {
    1: "Negligible",
    2: "Minor",
    3: "Moderate",
    4: "Major",
    5: "Catastrophic",
}

LIKELIHOOD_LEVELS = {
    1: "Rare",
    2: "Unlikely",
    3: "Possible",
    4: "Likely",
    5: "Almost certain",
}


def risk_band(score: int) -> str:
    """Map a 1–25 product to the explicitly provisional rating bands."""
    if score <= 4:
        return "Low"
    if score <= 9:
        return "Moderate"
    if score <= 16:
        return "High"
    return "Extreme"


def score_risk_output(output: dict[str, Any]) -> dict[str, Any]:
    """Compute each complete hazard's product and the maximum overall score."""
    scored: list[int] = []
    unscored = 0
    for hazard in output.get("hazards", []):
        severity = hazard.get("severity")
        likelihood = hazard.get("likelihood")
        if (isinstance(severity, int) and not isinstance(severity, bool) and 1 <= severity <= 5
                and isinstance(likelihood, int) and not isinstance(likelihood, bool) and 1 <= likelihood <= 5):
            hazard["score"] = severity * likelihood
            hazard["risk_rating"] = risk_band(hazard["score"])
            scored.append(hazard["score"])
        else:
            hazard["score"] = None
            hazard["risk_rating"] = None
            unscored += 1

    output["numeric_score"] = max(scored) if scored else None
    output["risk_rating"] = risk_band(output["numeric_score"]) if output["numeric_score"] is not None else None
    if not scored:
        output["assessment_status"] = "not_scored_missing_severity_or_likelihood"
    elif unscored:
        output["assessment_status"] = "provisional_partial_score_human_review_required"
    else:
        output["assessment_status"] = "provisional_score_human_review_required"
    output["scoring_method"] = "provisional_5x5_severity_x_likelihood_max_hazard"
    return output
