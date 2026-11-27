from __future__ import annotations

import re
from typing import Any, Mapping

AGENT_ROLES = (
    "investigation", "similar_case", "compliance", "risk", "mitigation",
    "critic", "verification",
)

# Query wording is generic and applies to any runway-incursion incident.
TOPIC_QUERIES = {
    "direct": "runway taxiway aircraft vehicle ATC clearance readback event chronology surface movement factual evidence",
    "historical": "historical runway incursion runway occupancy conflict aircraft vehicle takeoff landing taxi route clearance",
    "regulatory": "runway entry clearance hold short readback airport movement area air traffic control requirements procedures",
    "guidance": "runway incursion hazard safety barriers situational awareness signage surveillance communication preventive measures",
}


def _unique(items: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = str(item.get("chunk_id") or item.get("evidence_id") or item.get("document_id") or item.get("source_id") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        results.append(item)
        if limit is not None and len(results) == limit:
            break
    return results


# Look for actual clock-time transcript markers (e.g. 0639:13), not case-specific pages.
_TIMELINE_CLOCK = re.compile(r"\b(?:[01]\d|2[0-3])\d{2}:\s*[0-5]\d\b")
# Generic aviation actions that make a chronological passage useful for incident
# reconstruction. Terms are not tied to a particular airport or incident.
_ATC_ACTION = re.compile(
    r"\b(?:clear(?:ed|ance)?|take\s?off|abort(?:ed)?|go[ -]?around|"
    r"hold short|readback|land(?:ed|ing)?|runway crossing|taxi(?:ed|ing)?)\b",
    re.IGNORECASE,
)


def _event_relevance(item: dict[str, Any]) -> int:
    """Heuristic for choosing a useful chronology chunk, not a truth score."""
    excerpt = str(item.get("excerpt") or item.get("content") or item.get("text") or "")
    times = list(_TIMELINE_CLOCK.finditer(excerpt))
    if not times:
        return -1  # Prefer true clock-stamped transcript lines.
    actions = {m.group(0).lower() for m in _ATC_ACTION.finditer(excerpt)}
    conflict = bool(re.search(r"hold short|cross(?:ed|ing)?\s+runway|abort|reject(?:ed)?|go[ -]?around", excerpt, re.I))
    disputed = bool(re.search(r"incorrect|corrected|conflict|abort|reject(?:ed)?|go[ -]?around", excerpt, re.I))
    return min(len(times), 4) + 2 * min(len(actions), 5) + (10 if conflict else 0) + (6 if disputed else 0)



def _diverse_current(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep top-ranked current fact, then chronology, then distinct pages.

    The retrieved ranking still determines the candidates. This only avoids
    spending a small model's entire evidence budget on near-duplicate page hits.
    It never creates a passage or changes its source/citation identifiers.
    """
    if limit <= 0 or not items:
        return []

    chosen: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    used_pages: set[tuple[str, str]] = set()

    def page_key(item: dict[str, Any]) -> tuple[str, str] | None:
        page = item.get("page_number") or item.get("page")
        if page is None:
            return None
        return str(item.get("source_id") or item.get("document_id") or ""), str(page)

    def add(item: dict[str, Any]) -> bool:
        key = str(item.get("chunk_id") or item.get("evidence_id") or item.get("document_id") or item.get("source_id") or "")
        if not key or key in used_ids:
            return False
        chosen.append(item)
        used_ids.add(key)
        pg = page_key(item)
        if pg is not None:
            used_pages.add(pg)
        return True

    if limit >= 5:
        # A wider chronology packet preserves adjacent continuation chunks,
        # including untimed readbacks split at the original indexing boundary.
        # Only original retrieved passages and their IDs are sent onward.
        timed = sorted((r for r in items if _event_relevance(r) >= 0),
                       key=_event_relevance, reverse=True)
        for row in timed:
            if len(chosen) >= limit:
                break
            if page_key(row) in used_pages:
                continue
            add(row)
            for continuation in items:
                if len(chosen) >= limit:
                    break
                if page_key(row) is not None and page_key(continuation) == page_key(row):
                    add(continuation)
        for row in items:
            if len(chosen) >= limit:
                break
            add(row)
        def chronology_order(row):
            page = row.get("page_number") or row.get("page")
            return (str(row.get("source_id") or row.get("document_id") or ""),
                    int(page) if str(page).isdigit() else 10**9,
                    str(row.get("chunk_id") or row.get("evidence_id") or ""))
        chosen.sort(key=chronology_order)
        return chosen[:limit]

    add(items[0])
    if len(chosen) < limit:
        # A mere timestamp can describe an administrative briefing. Prefer a
        # time-stamped passage with operational clearances and radio actions.
        # Ties preserve the search engine's ranking, and selected IDs are intact.
        timed = [
            row for row in items[1:]
            if page_key(row) not in used_pages and _event_relevance(row) >= 0
        ]
        if timed:
            best = max(timed, key=_event_relevance)
            add(best)

    for item in items[1:]:
        if len(chosen) >= limit:
            break
        pg = page_key(item)
        if pg is not None and pg not in used_pages:
            add(item)

    for item in items[1:]:
        if len(chosen) >= limit:
            break
        add(item)
    return chosen[:limit]


def _case_uploads(items: list[dict[str, Any]], case_id: str) -> list[dict[str, Any]]:
    """User attachments are reported observations, not official factual findings."""
    source_counts: dict[str, int] = {}
    for row in items:
        source = str(row.get("source_id") or "").strip()
        if source:
            source_counts[source] = source_counts.get(source, 0) + 1
    uploads = []
    for row in items:
        if not row.get("evidence_id") or not row.get("excerpt"):
            continue
        source = str(row.get("source_id") or "").strip()
        # A unique source ID on a single submitted excerpt is a useful stable
        # passage citation (e.g. SFO17-SUMMARY). When a document ID is shared
        # by multiple excerpts, retain the server-generated evidence UUID.
        passage_id = source if source and source_counts.get(source) == 1 else None
        uploads.append({
            **row,
            **({"chunk_id": passage_id} if passage_id else {}),
            "source_id": row.get("source_id") or row["evidence_id"],
            "source_type": "user_submitted",
            "evidence_role": "reported",
            "incident_id": case_id,
            "evidence_status": "USER_REPORTED_UNVERIFIED",
        })
    return uploads


def _safe_query(narrative: str, topic: str, uploaded_evidence=None, incident_id=None) -> str:
    # Use event facts from submitted evidence, but never the unseen incident ID.
    # Keep the external embedding query bounded and avoid source URLs/metadata.
    from blackwing.rag.query_text import case_retrieval_query
    head = case_retrieval_query({
        "narrative": narrative,
        "evidence": uploaded_evidence or [],
        "source_incident_id": incident_id,
    }, max_evidence_chars=1000)
    return f"{topic} {head}"


def build_agent_packets(*, retriever: Any, narrative: str, case_id: str,
                        jurisdiction: str | None, incident_date: str | None,
                        documents: list[dict[str, Any]],
                        uploaded_evidence: list[dict[str, Any]] | None = None,
                        embedding_model: str = "BAAI/bge-small-en-v1.5",
                        limit_per_agent: int = 3,
                        method: str = "hybrid") -> dict[str, list[dict[str, Any]]]:
    """Route narrowly retrieved evidence to ALL seven existing agents.

    Retrieval stays in Python; the small Qwen model only sees the 1--3 passages
    relevant to its role. Backend can be local or IIT, as explicitly configured
    in the existing retriever. Never silently treat historical as current evidence.
    """
    if not 1 <= limit_per_agent <= 5:
        raise ValueError("limit_per_agent must be 1..5 to respect small Qwen context")
    if not case_id:
        raise ValueError("Case identity is required for source separation")
    uploads = _case_uploads(uploaded_evidence or [], case_id)

    def search(scope: str, limit: int, **kwargs: Any) -> list[dict[str, Any]]:
        result = retriever.search(
            _safe_query(narrative, TOPIC_QUERIES[scope], uploaded_evidence,
                        incident_id=case_id), documents,
            embedding_model=embedding_model, limit=limit, method=method,
            scope=scope, **kwargs,
        )
        return [dict(row) for row in result["results"]]

    direct_candidates = search("direct", 40, case_id=case_id, jurisdiction=jurisdiction,
                    incident_date=incident_date)
    historical = search("historical", 2, case_id=case_id,
                        incident_date=incident_date)
    # Defense in depth: a retriever must not return this incident as its own analogy.
    historical = [row for row in historical if str(row.get("incident_id") or "").upper() != case_id.upper()]
    for row in historical:
        row["evidence_role"] = "historical_analogy"
        row["caution"] = "OTHER INCIDENT: analogy only; never factual evidence for current case"
    rules = search("regulatory", 2, jurisdiction=jurisdiction,
                   incident_date=incident_date) if jurisdiction and incident_date else []
    # Several chunks of one guidance document would hide the others; keep up
    # to three distinct guidance sources.
    guidance, guidance_seen = [], set()
    for row in search("guidance", 20, incident_date=incident_date):
        source = str(row.get("source_id") or row.get("document_id") or "")
        if source and source not in guidance_seen:
            guidance_seen.add(source)
            guidance.append(row)
        if len(guidance) == 3:
            break
    for row in guidance:
        row["evidence_role"] = "general_guidance"
        row["caution"] = "Advisory, not a binding determination"
    direct = _diverse_current(direct_candidates, limit_per_agent)
    # Never drop user-supplied case documents because of the RAG top-k setting.
    current = _unique([*uploads, *direct], None)

    # One coherent, bounded evidence packet per agent; similar_case is
    # deterministic in the existing service. It must never receive case uploads.
    return {
        "investigation": current,

        "similar_case": _unique(
            historical, limit_per_agent
        ),

        "compliance": _unique(
            [*rules[:1], *current[:1], *guidance[:2]],
            limit_per_agent
        ),

        # Risk uses only input documents. Do not expose historical analogies or
        # general guidance to this agent's rating decision.
        "risk": _unique(uploads, limit_per_agent),

        "mitigation": _unique(
            [*guidance[:2], *current[:1]],
            limit_per_agent
        ),

        "critic": _unique(
            [*current[:max(2, limit_per_agent - 1)], *rules[:1]],
            limit_per_agent
        ),

        "verification": _unique(
            [*current[:max(2, limit_per_agent - 1)], *rules[:1]],
            limit_per_agent
        ),
    }

def packet_diagnostics(packets: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Structural check only; NOT retrieval Recall or model report accuracy."""
    return {
        "all_seven_roles_present": set(packets) == set(AGENT_ROLES),
        "source_counts": {role: len(packets.get(role, [])) for role in AGENT_ROLES},
        "sources": {role: [item.get("citation_label") or item.get("source_id")
                           for item in packets.get(role, [])] for role in AGENT_ROLES},
        "quality_claim": "NOT VERIFIED: evaluate with real BGE, CrossEncoder and complete gold relevance labels",
    }
