"""Citation identity and source-role checks shared by the five agent adapters.

A citation is a pointer, not proof that its text semantically supports a claim.
Only IDs present in an agent's supplied packet may be used. Historical sources
cannot substantiate current-incident facts, and user reports stay unverified.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

HISTORICAL_ROLES = {"historical_analogy", "historical", "similar_case"}
HISTORICAL_TYPES = {"official_final", "voluntary_asrs"}
DIRECT_ROLES = {"case_factual", "direct", "current_case", "reported"}
DIRECT_TYPES = {"official_factual", "official_preliminary", "user_submitted", "user_report"}


def passage_id(row: Mapping[str, Any]) -> str:
    return str(row.get("chunk_id") or row.get("evidence_id") or "").strip()


def source_id(row: Mapping[str, Any]) -> str:
    return str(row.get("source_id") or row.get("document_id") or row.get("evidence_id") or "").strip()


def is_historical(row: Mapping[str, Any]) -> bool:
    role = str(row.get("evidence_role") or "").casefold()
    kind = str(row.get("source_type") or "").casefold()
    return role in HISTORICAL_ROLES or kind in HISTORICAL_TYPES


def is_direct(row: Mapping[str, Any], incident_id: str | None = None) -> bool:
    """No unlabelled report is silently promoted to a current-case fact."""
    if is_historical(row):
        return False
    kind = str(row.get("source_type") or row.get("document_type") or "").casefold()
    role = str(row.get("evidence_role") or "").casefold()
    if kind not in DIRECT_TYPES:
        return False
    if kind in {"user_submitted", "user_report"}:
        return role in DIRECT_ROLES or bool(row.get("evidence_id"))
    if role not in {"case_factual", "direct", "current_case"}:
        return False
    row_incident = str(row.get("incident_id") or "").casefold()
    return bool(row_incident and (not incident_id or row_incident == incident_id.casefold()))


def citation_map(evidence: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, Mapping[str, Any]]]:
    """Map model-visible ID aliases to actual supplied passage IDs.

    Document ID alone is NOT accepted: one document could have many pages.
    Legacy zero-based chunk-index aliases are accepted only if unambiguous.
    """
    index: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for row in evidence:
        canonical = passage_id(row)
        if not canonical:
            continue
        aliases = {canonical}
        if row.get("document_id") is not None and row.get("chunk_index") is not None:
            aliases.add(f"{row['document_id']}:{row['chunk_index']}")
        for alias in aliases:
            if alias in index and index[alias][0] != canonical:
                index.pop(alias, None)  # ambiguous: fail closed
            else:
                index[alias] = (canonical, row)
    return index


def resolve(ids: Sequence[str], evidence: Sequence[Mapping[str, Any]], *,
            direct_only: bool = False, incident_id: str | None = None) -> tuple[list[str], list[str], list[str]]:
    """Return (canonical valid passage IDs, unique source IDs, rejected IDs)."""
    candidates = citation_map(evidence)
    clean: list[str] = []
    sources: list[str] = []
    rejected: list[str] = []
    for raw in ids:
        identifier = str(raw).strip()
        match = candidates.get(identifier)
        if not match or (direct_only and not is_direct(match[1], incident_id)):
            rejected.append(identifier)
            continue
        ref, row = match
        if ref not in clean:
            clean.append(ref)
        source = source_id(row)
        if source and source not in sources:
            sources.append(source)
    return clean, sources, rejected
