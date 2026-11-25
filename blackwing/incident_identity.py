"""Resolve explicit incident references without consulting evaluation reports."""
import re
from datetime import date

_REFERENCE = re.compile(r"(?<![A-Z0-9])(?:[A-Z]{3}\d{2}[A-Z]{2}\d{3})(?![A-Z0-9])", re.I)
_SYNTHETIC = re.compile(r"BW-SYN-|fictional|simulated training", re.I)


def resolve_incident_reference(narrative, explicit=None):
    """Use explicit metadata, or one unique supplied NTSB-style identifier.

    Repeated mentions count once. Ambiguous or synthetic inputs are not matched.
    This is routing metadata, never verification of an event or its claims.
    """
    if explicit and str(explicit).strip():
        return str(explicit).strip().upper()
    if _SYNTHETIC.search(narrative or ""):
        return None
    references = set(_REFERENCE.findall((narrative or "").upper()))
    return next(iter(references)) if len(references) == 1 else None


def declared_intake_metadata(narrative):
    """Read only an initial metadata header supplied by the user.

    These are retrieval hints, never independent evidence. Unknown, invalid or
    conflicting declarations are left unset; structured API fields take priority.
    Labels inside examples or the investigation body are never scanned.
    """
    if _SYNTHETIC.search(narrative or ''):
        return {}
    labels = {'SOURCE INCIDENT ID':'source_incident_id','INCIDENT DATE':'incident_date',
              'JURISDICTION':'jurisdiction','AIRPORT':'location'}
    found = {}
    started = False
    for line in (narrative or '').splitlines():
        line = line.strip()
        if not line:
            if started:
                break
            continue
        # Compact two-line prompts put the four routing fields on one line.
        # Accept only explicit known labels; never mine dates from case prose.
        matches = [re.fullmatch(r'(SOURCE INCIDENT ID|INCIDENT DATE|JURISDICTION|AIRPORT):[ \t]*(.*)',part.strip())
                   for part in line.split(';')]
        if not all(matches):
            break
        started = True
        for match in matches:
            key = labels[match[1]]
            found.setdefault(key,set()).add(match[2].strip())
    result = {}
    for key, values in found.items():
        if len(values) != 1:
            continue
        value = next(iter(values))
        if not value or value.casefold() in {'unknown','not supplied','unverified','n/a'}:
            continue
        if key == 'incident_date':
            try:
                if date.fromisoformat(value).isoformat() != value:
                    continue
            except ValueError:
                continue
        elif key == 'source_incident_id':
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{2,99}',value):
                continue
            value = value.upper()
        elif key == 'jurisdiction':
            aliases = {'US':'US','USA':'US','UNITED STATES':'US','IN':'IN','INDIA':'IN'}
            value = aliases.get(value.upper())
            if not value:
                continue
        elif len(value) > 300:
            continue
        result[key] = value
    return result
