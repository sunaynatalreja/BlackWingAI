"""Compile a preliminary investigation report from accepted case findings."""

from __future__ import annotations

from html import escape
from io import BytesIO
import re


REPORT_SECTIONS = {
    "overview": "Incident overview",
    "chronology": "Event chronology",
    "evidence": "Evidence and contributing-factor analysis",
    "hazards": "Hazards and risk assessment",
    "regulatory": "Regulatory assessment",
    "mitigation": "Preventive measures and mitigation plan",
}
LIMITATIONS = (
    "This preliminary report records findings accepted for this case workspace. "
    "Official causes, severity classifications, compliance decisions and operational actions "
    "require qualified review and the appropriate authority.",
    "The overview metadata comes from case intake. Each finding retains its cited evidence and review record.",
    "Chronology entries follow the saved finding order. Check event times and time zones against the source records.",
)
SCOPE_LABELS = {
    "direct": "Direct case evidence",
    "submitted": "Submitted evidence",
    "reported": "Reported intake",
    "analogy": "Historical comparison",
    "comparison": "Historical comparison",
    "rule": "Candidate rule",
    "regulatory": "Candidate rule",
    "candidate_guidance": "Candidate guidance",
}


def _field(record: dict, key: str) -> str:
    value = record.get(key)
    return value.strip() if isinstance(value, str) else ""


def compile_report(incident: dict, findings: list[dict]) -> dict:
    """Return a report snapshot with only reviewed, traceable findings.

    Citations are the persisted excerpts reviewed alongside each claim. The caller
    supplies case metadata and stored findings. Compilation makes no model calls.
    """
    metadata = {key: _field(incident, key) or "Unspecified"
                for key in ("case_id", "airport", "event_date", "jurisdiction")}
    sections = {key: [] for key in REPORT_SECTIONS}
    findings_seen = set()
    for finding in findings:
        if not isinstance(finding, dict) or finding.get("status") != "accepted":
            continue
        section = _field(finding, "section")
        finding_id = _field(finding, "finding_id")
        claim = _field(finding, "claim")
        reviewer = _field(finding, "reviewed_by")
        reviewed_at = _field(finding, "reviewed_at")
        case_id = _field(finding, "case_id")
        citations = finding.get("citations")
        if section not in sections or not all((finding_id, claim, reviewer, reviewed_at)):
            continue
        if finding_id in findings_seen or (case_id and case_id != metadata["case_id"]):
            continue
        if not isinstance(citations, list) or not citations or not all(
            isinstance(citation, dict) and all(_field(citation, key) for key in ("source_id", "title", "excerpt"))
            for citation in citations
        ):
            continue
        related = finding.get("related_finding_ids")
        sections[section].append({
            "finding_id": finding_id,
            "claim": claim,
            "reviewed_by": reviewer,
            "reviewed_at": reviewed_at,
            "kind": _field(finding, "kind"),
            "event_time": _field(finding, "event_time"),
            "uncertainty": _field(finding, "uncertainty"),
            "potential_consequence": _field(finding, "potential_consequence"),
            "existing_controls": [text for text in (finding.get("existing_controls") or [])
                                  if isinstance(text, str) and text.strip()]
                                 if isinstance(finding.get("existing_controls"), list) else [],
            "related_finding_ids": list(dict.fromkeys(value for value in related if isinstance(value, str)))
            if isinstance(related, list) else [],
            "citations": citations,
        })
        findings_seen.add(finding_id)

    sources = {}
    for rows in sections.values():
        for row in rows:
            refs = []
            for citation in row.pop("citations"):
                page = citation.get("page")
                if isinstance(page, (str, int)) and not isinstance(page, bool):
                    page = str(page).strip()
                else:
                    page = ""
                source = {key: _field(citation, key) for key in ("source_id", "title", "chunk_id", "scope", "excerpt", "locator")}
                source["page"] = page
                key = tuple(source.values())
                reference = sources.setdefault(key, {"number": len(sources) + 1, **source})
                refs.append(reference["number"])
            row["citation_refs"] = list(dict.fromkeys(refs))
            row["related_finding_ids"] = [item for item in row["related_finding_ids"] if item in findings_seen]

    return {
        "incident": metadata,
        "sections": [{"key": key, "title": title, "findings": sections[key]}
                     for key, title in REPORT_SECTIONS.items()],
        "sources": list(sources.values()),
        "accepted_count": len(findings_seen),
        "limitations": list(LIMITATIONS),
    }


def _markup(value: str) -> str:
    """Escape untrusted plain text before sending it to ReportLab's paragraph parser."""
    dashes = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(value).translate(dashes))
    return escape(text)


def render_report_pdf(report: dict) -> bytes:
    """Render the compiled report as an A4 PDF, with links to in-document references."""
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    navy = HexColor("#122539")
    teal = HexColor("#13636F")
    gray = HexColor("#465A6E")
    line = HexColor("#D2DEE6")
    pale = HexColor("#EEF5F6")

    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=10.5, leading=15,
                          alignment=TA_LEFT, textColor=navy, spaceAfter=6)
    heading = ParagraphStyle("Heading", parent=body, fontName="Helvetica-Bold", fontSize=13,
                             leading=18, textColor=teal, spaceBefore=16, spaceAfter=5, keepWithNext=True)
    title = ParagraphStyle("Title", parent=body, fontName="Helvetica-Bold", fontSize=24,
                           leading=29, spaceBefore=3, spaceAfter=15)
    small = ParagraphStyle("Small", parent=body, fontSize=8.5, leading=12, textColor=gray)
    finding_text = ParagraphStyle("Finding", parent=body, spaceAfter=3, keepWithNext=True)
    finding_detail = ParagraphStyle("Finding detail", parent=small, spaceAfter=8)
    source_title = ParagraphStyle("Source title", parent=body, fontName="Helvetica-Bold",
                                  spaceBefore=5, spaceAfter=2, keepWithNext=True)
    source_meta = ParagraphStyle("Source metadata", parent=small, spaceAfter=4, keepWithNext=True)
    source_excerpt = ParagraphStyle("Source excerpt", parent=small, leftIndent=10, spaceAfter=9)

    stream = BytesIO()
    document = SimpleDocTemplate(stream, pagesize=A4, leftMargin=46, rightMargin=46,
                                 topMargin=60, bottomMargin=55,
                                 title="BLACKWING AI - Preliminary Investigation Report",
                                 author="BLACKWING AI Copilot",
                                 subject="Report compiled from case findings accepted for review")

    def draw_frame(canvas, page):
        width, height = A4
        canvas.setStrokeColor(line)
        canvas.setFillColor(teal)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(page.leftMargin, height - 33, "BLACKWING AI  /  INVESTIGATION COPILOT")
        canvas.line(page.leftMargin, height - 42, width - page.rightMargin, height - 42)
        canvas.line(page.leftMargin, 42, width - page.rightMargin, 42)
        canvas.setFillColor(gray)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(page.leftMargin, 28, "Preliminary investigation report")
        canvas.drawRightString(width - page.rightMargin, 28, f"Page {page.page}")

    story = [Paragraph("Preliminary Investigation Report", title)]
    metadata = report["incident"]
    labels = (("Case identifier", "case_id"), ("Airport", "airport"),
              ("Event date", "event_date"), ("Jurisdiction", "jurisdiction"))
    width = document.width - 12
    table = Table([[Paragraph(f"<b>{label}</b><br/>{_markup(metadata[key])}", body)
                    for label, key in labels]], colWidths=[width * fraction for fraction in (.36, .18, .23, .23)])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), pale),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
    ]))
    count = report["accepted_count"]
    story.extend([table, Spacer(1, 8), Paragraph(
        f"{count} accepted {'finding' if count == 1 else 'findings'} from the case review record.", small
    )])
    for index, section in enumerate(report["sections"], 1):
        story.append(Paragraph(f"{index}. {_markup(section['title'])}", heading))
        if not section["findings"]:
            story.append(Paragraph("No accepted findings recorded.", small))
        for finding in section["findings"]:
            refs = " ".join(f'<link href="#source-{number}" color="#13636F">[{number}]</link>'
                            for number in finding["citation_refs"])
            event_time = finding["event_time"] if section["key"] == "chronology" else ""
            prefix = f"<b>{_markup(event_time)}</b> " if event_time else ""
            story.append(Paragraph(f"{prefix}{_markup(finding['claim'])} {refs}", finding_text))
            details = [
                f"<b>Finding</b> {_markup(finding['finding_id'])}",
                f"<b>Accepted by</b> {_markup(finding['reviewed_by'])}",
                _markup(finding["reviewed_at"]),
            ]
            if finding["kind"]:
                details.append(f"<b>Type</b> {_markup(finding['kind'].replace('_', ' ').title())}")
            if finding["related_finding_ids"]:
                related = ", ".join(_markup(item) for item in finding["related_finding_ids"])
                details.append(f"<b>Related findings</b> {related}")
            if finding.get("potential_consequence"):
                details.append(f"<b>Potential consequence</b> {_markup(finding['potential_consequence'])}")
            if finding.get("existing_controls"):
                controls = " | ".join(_markup(control) for control in finding["existing_controls"])
                details.append(f"<b>Existing controls cited</b> {controls}")
            if finding["uncertainty"]:
                details.append(f"<b>Uncertainty</b> {_markup(finding['uncertainty'])}")
            story.append(Paragraph(" &nbsp;|&nbsp; ".join(details[:3]) +
                                   ("<br/>" + "<br/>".join(details[3:]) if len(details) > 3 else ""),
                                   finding_detail))
    story.append(Paragraph("Scope and limitations", heading))
    for note in report["limitations"]:
        story.append(Paragraph(_markup(note), body))
    story.append(Paragraph("Evidence references", heading))
    if not report["sources"]:
        story.append(Paragraph("No references are associated with accepted findings.", small))
    for source in report["sources"]:
        number = source["number"]
        story.append(Paragraph(f'<a name="source-{number}"/>[{number}] {_markup(source["title"])}', source_title))
        details = [source["source_id"]]
        if source["locator"]:
            details.append(source["locator"])
        elif source["page"]:
            details.append(f"Page {source['page']}")
        if source["chunk_id"] and source["chunk_id"] != source["source_id"]:
            details.append(source["chunk_id"])
        if source["scope"]:
            details.append(SCOPE_LABELS.get(source["scope"], source["scope"].replace("_", " ").capitalize()))
        story.append(Paragraph(" &nbsp;|&nbsp; ".join(_markup(item) for item in details), source_meta))
        story.append(Paragraph(_markup(source["excerpt"]), source_excerpt))
    document.build(story, onFirstPage=draw_frame, onLaterPages=draw_frame)
    return stream.getvalue()
