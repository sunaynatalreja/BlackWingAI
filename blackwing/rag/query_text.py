"""Case content for retrieval; keep strategy instructions out of search features."""
import re


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
    lines=[]
    for line in text.splitlines():
        if re.match(r'\s*(?:CASE ID|PROMPT STRATEGY|INVESTIGATION (?:REQUEST|REQUIREMENTS)|SUPPLIED SYNTHETIC ACCOUNTS)\b',line,re.I):
            continue
        line=re.sub(r'\b(?:SOURCE INCIDENT ID|INCIDENT DATE|JURISDICTION|AIRPORT):\s*','',line,flags=re.I)
        lines.append(line)
    return ' '.join(' '.join(lines).split())
