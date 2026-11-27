"""Deterministic, deduplicated, provenance-bound historical-case comparator.

Similar-case similarity is a *candidate analogy*, never a factual finding
about the current incident. No LLM call or extra IIT tokens are required.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from blackwing.agents.evidence_utils import source_id
from blackwing.agents.similar_case.schema import SimilarCaseMatch, SimilarCaseOutput

_TERMS = re.compile(r"[a-z0-9]{3,}", re.I)
_STOP = {"runway", "aircraft", "airport", "reported", "incident", "the", "and", "with", "from", "that", "when"}
_STOP.update({'for','only','this','not','few','report','reports','was','were','are','has','had',
              'have','but','its','their','they','into','any','can','may','would','which','what',
              'use','using','obtain','existing','supplied','source','sources','evidence','case',
              'prompt','strategy','normal','schema','gold','human','review','verification'})


def _features(s: str) -> set[str]:
    return set(_TERMS.findall(s.lower())) - _STOP


def build_output(narrative: str, evidence: Sequence[Mapping[str, Any]], *,
                 current_incident_id: str | None = None) -> dict[str, Any]:
    reports: dict[str, Mapping[str, Any]] = {}
    exact_records: list[Mapping[str, Any]] = []
    for item in evidence:
        # Exact case identity is a metadata match against an independently
        # identified current-case source. Keep it separate from analogies.
        if (current_incident_id
                and str(item.get("incident_id") or "").casefold() == current_incident_id.casefold()
                and str(item.get("source_type") or "").casefold() in {"official_factual", "official_preliminary"}
                and str(item.get("evidence_role") or "").casefold() in {"case_factual", "direct", "current_case"}):
            exact_records.append(item)
        dtype = str(item.get("document_type") or "").casefold()
        if dtype not in {"incident_report", "voluntary_incident_report"}:
            continue
        if dtype == "voluntary_incident_report" and (
            item.get("source_type") != "voluntary_asrs"
            or item.get("evidence_role") != "historical_analogy"
        ):
            continue
        case_id = str(item.get("incident_id") or "")
        if current_incident_id and case_id.casefold() == current_incident_id.casefold():
            continue
        source = source_id(item)
        if not source:
            continue
        old = reports.get(source)
        if old is None or (not old.get("historical_findings") and item.get("historical_findings")):
            reports[source] = item

    from blackwing.rag.query_text import case_query_text
    targets = _features(case_query_text(narrative))
    scored: list[tuple[float, str, Mapping[str, Any], list[str]]] = []
    for ref, item in reports.items():
        text = " ".join(str(item.get(k) or "") for k in ("title", "excerpt", "content", "comparison_notes"))
        overlapping = sorted(targets & _features(text))
        if len(overlapping) < 2:
            continue
        # Retrieving a case is not evidence of similarity; require at least two
        # nontrivial shared content terms, or state insufficiency.
        score = len(overlapping) / (len(targets) + 1)
        scored.append((score, ref, item, overlapping))
    scored.sort(key=lambda t: (-t[0], t[1]))

    matches = []
    for score, ref, item, features in scored[:3]:
        notes = list(item.get("comparison_notes") or [])
        if item.get("source_type") == "voluntary_asrs":
            notes.append("Voluntary ASRS self-report: historical analogy only; not verified official causal findings or current-case evidence.")
        if not features:
            notes.append("No specific shared operational feature was established from available excerpts.")
        if not item.get("incident_date"):
            notes.append("Historical incident date not verified from supplied metadata.")
        matches.append(SimilarCaseMatch(
            source_id=ref,
            title=str(item.get("title") or "Historical incident report"),
            source_url=item.get("source_url"),
            similarity=("Potential shared terms from supplied excerpts: " + ", ".join(features[:6]) + "."
                        if features else "Retrieved historical record; specific analogy not established."),
            differences_or_unknowns=[str(x) for x in notes][:8],
            source_findings=[str(x) for x in (item.get("historical_findings") or [])][:5],
        ))
    if exact_records:
        status = "existing_case_match"
        assessment = "An incident identifier matches an independently identified current-case source."
    elif matches:
        status = "unseen_case_with_similar_historical_cases"
        assessment = f"{len(matches)} retrieved historical source(s); each analogy is a candidate only."
    else:
        status = "unseen_case_no_historical_match"
        assessment = "No historical analogy with at least two shared case-content terms was established from supplied excerpts."
    return SimilarCaseOutput(
        case_status=status, matches=matches, assessment=assessment,
        factors_to_verify=["Verify that the matched source covers this incident, not a related occurrence."] if exact_records else
            (["Check which operational conditions genuinely match before transferring any lessons."] if matches else []),
        limitations=[
            "A retrieved historical case does not establish the cause or contributing factors of this event.",
            "Source authenticity, dates and comparisons require a human investigator.",
        ],
    ).model_dump(mode="json")
