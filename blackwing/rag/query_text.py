"""Case content for retrieval; keep strategy instructions out of search features."""
import re


_INTAKE_METADATA = re.compile(
    r"\b(?:SOURCE\s+INCIDENT\s+ID|INCIDENT\s+DATE|JURISDICTION|AIRPORT)\s*:\s*"
    r"[^;\n|]*(?=;|\||\n|$)", re.I,
)


def case_query_text(narrative):
    text=str(narrative or '')
    # The short fictional paragraph contains the case, not the instruction header.
    synthetic=re.search(r'^ORIGINAL FICTIONAL[^\n]+',text,re.M|re.I)
    if synthetic:
        text=synthetic[0]
    else:
        requested=re.search(r'(?:\|\s*REQUEST:|^REQUESTED CASE\s*\n)([^\n]+)',text,re.M|re.I)
        if requested:
            text=requested[1]
        else:
            scenario=re.search(r'^SCENARIO:\s*\n?([^\n]+)',text,re.M|re.I)
            if scenario:
                text=scenario[1]
    text=re.split(r'\b(?:Use the normal|Use the existing|These intake details|Intake metadata|'
                  r'VERIFICATION FOCUS:|STRATEGY GUIDANCE|KNOWN INPUT LIMITS|'
                  r'No event date or country/jurisdiction|These are unverified synthetic accounts)(?=\s|$)',text,maxsplit=1,flags=re.I)[0]
    # Intake metadata may be semicolon-separated on one line. Remove each
    # complete field value, not just its label, so an unseen ID never becomes
    # a lexical or embedding retrieval token.
    text = _INTAKE_METADATA.sub(" ", text)
    lines=[]
    for line in text.splitlines():
        if re.match(r'\s*(?:CASE ID|PROMPT STRATEGY|INVESTIGATION (?:REQUEST|REQUIREMENTS)|SUPPLIED SYNTHETIC ACCOUNTS)\b',line,re.I):
            continue
        lines.append(line)
    return re.sub(r'[;|]+', ' ', ' '.join(' '.join(lines).split())).strip()


def case_retrieval_query(case, *, max_evidence_chars=1600):
    """Build a retrieval query from incident facts, never from its incident ID.

    Uploaded case evidence can contain the concrete aircraft, sequence, and
    operational details absent from a short intake prompt. Those contents may
    guide retrieval, while uploaded evidence itself is never a corpus target.
    """
    narrative = case_query_text(case.get("narrative") or "")
    incident_id = str(case.get("source_incident_id") or "").strip()
    if incident_id:
        narrative = re.sub(re.escape(incident_id), " ", narrative, flags=re.I)

    evidence_parts = []
    for row in case.get("evidence") or []:
        text = str(row.get("excerpt") or row.get("content") or "")
        if not text:
            continue
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"\[[A-Z]{2,}[A-Z0-9-]*\d[A-Z0-9-]*\]", " ", text, flags=re.I)
        if incident_id:
            text = re.sub(re.escape(incident_id), " ", text, flags=re.I)
        retained = []
        for line in text.splitlines():
            line = line.strip()
            if not line or re.match(
                r"^(?:SOURCED? FACTUAL INPUT|SOURCE\s+\S+\s*:|CASE\s+ID\s*:|"
                r"INCIDENT\s+ID\s*:|SOURCE\s+SFO\S*)",
                line, re.I,
            ):
                continue
            retained.append(line)
        cleaned = " ".join(" ".join(retained).split())
        if cleaned:
            evidence_parts.append(cleaned)
    evidence_text = " ".join(evidence_parts)[:max_evidence_chars]
    # Put source facts before the short routing request; BGE query clipping is
    # prefix-based, so this preserves the most useful event details.
    return " ".join(part for part in (evidence_text, narrative) if part).strip()
