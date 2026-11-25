"""Conservative conflict detection for incident-specific evidence.

This is a rejection check, not proof that a source belongs to a case.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Mapping

_FLIGHT = re.compile(r"\b(?:flight\s+)?([A-Z]{3})\s*[- ]?\s*(\d{2,4})\b", re.I)
_DATE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b", re.I)


def _identity(text: str) -> tuple[set[str], set[str]]:
    flights = {a.upper() + b for a, b in _FLIGHT.findall(text)}
    dates = set()
    for value in _DATE.findall(text):
        for fmt in ("%Y-%m-%d", "%d %B %Y"):
            try:
                dates.add(datetime.strptime(value, fmt).date().isoformat())
                break
            except ValueError:
                pass
    return flights, dates


def conflicts_with_case(row: Mapping[str, Any], narrative: str, incident_date: str | None = None) -> bool:
    """Reject explicit contradictions; missing identifiers remain unverified.

    Dates in an excerpt may refer to unrelated events, so only explicit incident
    metadata and dates in a report title are compared.
    """
    target_flights, target_dates = _identity(narrative)
    if incident_date:
        target_dates.add(str(incident_date)[:10])
    title = str(row.get("title") or "")
    source_flights, title_dates = _identity(title)
    # Flight identifiers in the passage can identify otherwise untitled uploads.
    excerpt_flights, _ = _identity(str(row.get("excerpt") or row.get("content") or ""))
    source_flights |= excerpt_flights
    source_dates = {str(row["incident_date"])[:10]} if row.get("incident_date") else title_dates
    return bool((target_flights and source_flights and target_flights.isdisjoint(source_flights)) or
                (target_dates and source_dates and target_dates.isdisjoint(source_dates)))
