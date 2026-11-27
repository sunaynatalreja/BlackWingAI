
"""Preliminary FAA airport risk-matrix reference calculator.

Reference: FAA AC 150/5200-37A, Figure 4-5.
Not an approved operational risk assessment.
"""

SEVERITY = {
    1: "catastrophic",
    2: "hazardous",
    3: "major",
    4: "minor",
    5: "minimal",
}

LIKELIHOOD = {
    "A": "frequent",
    "B": "probable",
    "C": "remote",
    "D": "extremely_remote",
    "E": "extremely_improbable",
}

# Columns correspond to severity 1, 2, 3, 4, 5.
FAA_REFERENCE_MATRIX = {
    "A": ("high", "high", "high", "medium", "low"),
    "B": ("high", "high", "high", "medium", "low"),
    "C": ("high", "high", "medium", "medium", "low"),
    "D": ("high", "medium", "medium", "low", "low"),
    "E": ("special_review", "medium", "low", "low", "low"),
}


def assess_reference_risk(
    severity=None,
    likelihood=None,
    *,
    evidence_ids=None,
    severity_basis="",
    likelihood_basis="",
):
    """Calculate a reference risk band only when supporting inputs exist."""

    if not (
        severity is not None
        and likelihood is not None
        and evidence_ids
        and severity_basis.strip()
        and likelihood_basis.strip()
    ):
        return {
            "status": "unassessed_insufficient_inputs",
            "risk_index": None,
            "risk_level": None,
            "human_review_required": True,
        }

    if severity not in SEVERITY:
        raise ValueError("Severity must be an integer from 1 to 5")

    if likelihood not in LIKELIHOOD:
        raise ValueError("Likelihood must be A, B, C, D or E")

    level = FAA_REFERENCE_MATRIX[likelihood][severity - 1]

    return {
        "status": "provisional_reference_only",
        "risk_index": f"{likelihood}{severity}",
        "severity": SEVERITY[severity],
        "likelihood": LIKELIHOOD[likelihood],
        "risk_level": level,
        "human_review_required": True,
        "matrix_reference": "FAA_AC_150_5200_37A_Figure_4_5",
        "limitations": (
            "Reference assessment only. Inputs and risk acceptance "
            "must be reviewed by qualified aviation safety personnel."
        ),
    }
