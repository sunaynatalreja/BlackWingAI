"""Deterministic, source-deduplicated comparison for retrieved historical cases."""

from __future__ import annotations

from typing import Any

from blackwing.agents.similar_case.schema import SimilarCaseMatch, SimilarCaseOutput


def build_output(narrative: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one grounded match per source, without asking a small model to cite it."""
    reports: dict[str, dict[str, Any]] = {}
    for item in evidence:
        if str(item.get("document_type", "")).casefold() != "incident_report":
            continue
        source_id = item.get("source_id")
        if not source_id:
            continue
        current = reports.setdefault(source_id, item)
        # Prefer the chunk carrying curated facts, but retain the first source metadata.
        if item.get("historical_findings") and not current.get("historical_findings"):
            reports[source_id] = {**current, **item}

    matches: list[SimilarCaseMatch] = []
    exact_reference_found = False
    for source_id, item in reports.items():
        source_text = " ".join(str(item.get(key, "")) for key in ("title", "excerpt", "source_id")).casefold()
        # Exact matching is intentionally conservative; retrieval similarity alone
        # must never classify an incident as an already investigated case.
        reference_tokens = [token for token in narrative.split() if token.upper().startswith(("JFK-", "DCA", "NTSB-"))]
        if reference_tokens and any(token.casefold().strip(".,:;") in source_text for token in reference_tokens):
            exact_reference_found = True
        findings = item.get("historical_findings") or []
        notes = item.get("comparison_notes") or []
        matches.append(SimilarCaseMatch(
            source_id=source_id,
            title=str(item.get("title", "Historical incident report")),
            source_url=item.get("source_url"),
            similarity=(
                "The curator-identified analogy is a runway conflict involving takeoff operations "
                "and another aircraft movement on the same runway. See the material differences below."
            ),
            differences_or_unknowns=[str(note) for note in notes],
            source_findings=[str(finding) for finding in findings],
        ))

    if exact_reference_found:
        status = "existing_case_match"
        assessment = "A source containing the supplied incident reference was retrieved; verify that it describes the same event before reusing any findings."
    elif matches:
        status = "unseen_case_with_similar_historical_cases"
        assessment = (
            "The incoming incident reference is not present in the retrieved reports. "
            f"{len(matches)} distinct historical report(s) provide analogy only. Their official causes apply to those historical events, not this case."
        )
    else:
        status = "unseen_case_no_historical_match"
        assessment = "No historical incident report matched the current retrieval query; this case remains unseen in the indexed corpus."

    return SimilarCaseOutput(
        case_status=status,
        matches=matches,
        assessment=assessment,
        factors_to_verify=(
            [
                "Controller traffic awareness and coordination at the time of the crossing.",
                "Flight-crew surface-navigation and runway-clearance cross-checks.",
                "Timing and effectiveness of surface-surveillance alerts and controller intervention.",
            ] if matches else []
        ),
        limitations=[
            "The historical report is an analogy, not evidence of what caused the incoming event.",
            "Verify all source details and current-case facts with primary records and a qualified investigator.",
        ],
    ).model_dump(mode="json")
