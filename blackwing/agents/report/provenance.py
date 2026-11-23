"""Report provenance ceilings; citation existence does not prove entailment."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from blackwing.agents.evidence_utils import is_direct, is_historical


def collect_report_evidence(shared, agent_packets=None):
    """Retain the provenance of passages already supplied to upstream agents.

    Conflicting provenance for the same passage is retained, not overwritten.
    This reads existing packets only; it performs no retrieval or model call.
    """
    result, seen = [], set()
    rows = list(shared or [])
    for packet in (agent_packets or {}).values():
        rows.extend(packet)
    for row in rows:
        key = tuple(str(row.get(k) or '') for k in (
            'chunk_id', 'evidence_id', 'source_id', 'source_type', 'document_type',
            'evidence_role', 'evidence_status', 'incident_id'))
        if key not in seen:
            seen.add(key)
            result.append(deepcopy(row))
    return result


def ground_report_point(point: Mapping[str, Any], evidence, *, incident_id=None):
    """Never let report wording strengthen the provenance of its citations."""
    result = deepcopy(dict(point))
    if result.get('classification') not in {'documented_fact', 'reported_observation'}:
        return result
    by_id = {}
    for row in evidence:
        for key in ('chunk_id', 'evidence_id'):
            if row.get(key):
                by_id.setdefault(str(row[key]), []).append(row)
    refs = result.get('evidence_ids') or []
    # A source/document identifier without a supplied passage is not enough.
    if not refs or any(ref not in by_id for ref in refs):
        result['classification'] = 'verification_gap'
        return result
    rows = [row for ref in refs for row in by_id[ref]]
    if any(not is_direct(row, incident_id) for row in rows):
        result['classification'] = ('historical_analogy' if all(is_historical(r) for r in rows)
                                    else 'verification_gap')
        return result
    unverified = any(
        str(row.get('source_type') or '').casefold() in {'user_submitted', 'user_report'}
        or 'unverified' in str(row.get('evidence_status') or '').casefold()
        for row in rows)
    if unverified:
        result['classification'] = 'reported_observation'
    return result
