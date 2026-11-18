"""Human-readable, citation-transparent Markdown view of the existing report JSON.

This is presentation only; it never turns model output into verified facts.
Both JSON and Markdown exports remain preliminary until human review.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

_EMAIL = re.compile(r"(?<![\w.])([\w.+-]+@[\w.-]+\.[a-zA-Z]{2,})(?![\w.])")
_PHONE = re.compile(r"(?<!\d)(?:\+?1[- .]?)?\(?\d{3}\)?[- .]?\d{3}[- .]?\d{4}(?!\d)")


def _safe(value: Any) -> str:
    # Contact redaction is a defense in depth, not complete PII de-identification.
    s = str(value or "").strip().replace("\r", " ").replace("\n", " ")
    s = _EMAIL.sub("[EMAIL REDACTED]", s)
    s = _PHONE.sub("[PHONE REDACTED]", s)
    return s.replace("`", "'").replace("|", "\\|")


def safe_report_text(value: Any) -> str:
    """Apply the report view's contact redaction before presenting user text."""
    return _safe(value)


def _ids(value: Any) -> str:
    refs = [str(x) for x in (value or []) if str(x).strip()]
    return " [passages: " + ", ".join(_safe(x) for x in refs) + "]" if refs else " [source not established]"


def render_markdown_report(container: Mapping[str, Any]) -> str:
    r = container.get("report") or container
    generated = r.get("generated_report")
    if generated:
        return render_agent_report_markdown(container, generated)
    outputs = r.get("agent_outputs") or {}
    inv = outputs.get("investigation") or {}
    sim = outputs.get("similar_case") or {}
    comp = outputs.get("compliance") or {}
    risk = outputs.get("risk") or {}
    mit = outputs.get("mitigation") or {}
    critic = outputs.get("critic") or {}
    ver = outputs.get("verification") or {}
    lines: list[str] = [
        "# BLACKWING AI — Preliminary Runway-Incursion Investigation",
        "", "> **AI-assisted draft. Not a factual determination, legal opinion or operational directive. Human verification required.**",
        "", f"**Case:** `{_safe(r.get('case_id') or container.get('case_id'))}`  ",
        f"**Model:** `{_safe(r.get('model'))}`  ",
        f"**Report status:** `{_safe(r.get('status'))}`", "",
        "## 1. Event Summary", "", _safe(inv.get("incident_summary")) or "Not established.", "",
        "## 2. Timeline and Evidence Review", "",
    ]
    for ev in inv.get("timeline_events", []):
        lines.append(f"- {_safe(ev.get('time_reference') or 'Time unknown')}: "
                     f"{_safe(ev.get('event'))}{_ids(ev.get('evidence_ids'))} "
                     f"(**{_safe(ev.get('evidence_state'))}**)")
    if not inv.get("timeline_events"):
        lines.append("- No evidence-grounded timeline available.")
    for heading, field in (("Documented facts", "documented_facts"),
                           ("Reported observations", "reported_observations"),
                           ("Hypotheses and possible factors", "hypotheses"),
                           ("Potential contributing factors", "contributing_factors")):
        lines += ["", f"### {heading}", ""]
        for claim in inv.get(field, []):
            lines.append(f"- {_safe(claim.get('text'))}{_ids(claim.get('evidence_ids'))} "
                         f"(**{_safe(claim.get('evidence_state'))}**)")
        if not inv.get(field):
            lines.append("- None established.")
    lines += ["", "### Information gaps", ""]
    lines += ["- " + _safe(x) for x in inv.get("unknowns", [])] or ["- None recorded; source review still required."]

    lines += ["", "## 3. Historical Case Comparisons", "", _safe(sim.get("assessment")), ""]
    for match in sim.get("matches", []):
        lines.append(f"- **{_safe(match.get('title'))}** (source: `{_safe(match.get('source_id'))}`): "
                     + _safe(match.get("similarity")) + " **Analogy only.**")
    if not sim.get("matches"):
        lines.append("- No distinct historical comparison supplied.")

    lines += ["", "## 4. Regulatory and Procedural Applicability", "",
              f"**Status:** {_safe(comp.get('overall_applicability') or 'not_assessed')}", ""]
    for check in comp.get("checks", []):
        lines.append(f"- {_safe(check.get('source_title') or check.get('source_id'))}: "
                     f"{_safe(check.get('applicability_status'))}; "
                     f"{_safe(check.get('rationale'))}{_ids(check.get('evidence_ids'))}")
    if not comp.get("checks"):
        lines.append("- No eligible rule was assessed; no violation is asserted.")
    for missing in comp.get("missing_context", []):
        lines.append("- **To verify:** " + _safe(missing))

    lines += ["", "## 5. Hazards and Risk", "",
              f"**Scoring:** {_safe(risk.get('assessment_status'))}. "
              "No numerical score without an approved risk matrix.", ""]
    for h in risk.get("hazards", []):
        lines.append(f"- **{_safe(h.get('hazard_id'))}**: {_safe(h.get('description'))}"
                     f"{_ids(h.get('evidence_ids'))}")
        if h.get("existing_controls"):
            lines.append("  - Potential controls: " + "; ".join(_safe(x) for x in h["existing_controls"]))
        if h.get("control_gaps"):
            lines.append("  - Control questions: " + "; ".join(_safe(x) for x in h["control_gaps"]))
    if not risk.get("hazards"):
        lines.append("- No evidence-grounded hazards established.")
    if risk.get("risk_notes"):
        lines += ["", _safe(risk["risk_notes"])]

    lines += ["", "## 6. Proposed Preventive and Corrective Measures", "",
              "These are options for qualified review, not operational instructions.", ""]
    for rec in mit.get("recommendations", []):
        lines.append(f"- **{_safe(rec.get('action'))}** — "
                     f"linked hazards: {', '.join(_safe(x) for x in rec.get('hazard_ids', [])) or 'none'}; "
                     f"priority: {_safe(rec.get('priority'))}; "
                     f"indicator: {_safe(rec.get('effectiveness_indicator') or 'to be defined')}.")
    if not mit.get("recommendations"):
        lines.append("- No currently traceable recommendation; request risk/incident evidence.")
    for assumption in mit.get("assumptions", []):
        lines.append("- **Assumption:** " + _safe(assumption))

    lines += ["", "## 7. Quality Review and Verification", ""]
    for issue in critic.get("issues", []):
        lines.append(f"- **{_safe(issue.get('issue_type'))}** "
                     f"({_safe(issue.get('affected_output') or 'general')}): {_safe(issue.get('description'))}")
    for claim in ver.get("claims_needing_source", []):
        lines.append("- **Requires source verification:** " + _safe(claim))
    for item in ver.get("unsupported_conclusions", []):
        lines.append("- **Unsupported / provisional:** " + _safe(item))
    lines += ["", f"**Completeness:** {_safe(ver.get('completeness_status'))}  ",
              "**Human approval required:** Yes", "", "## 8. Limitations", ""]
    lines += ["- " + _safe(x) for x in r.get("limitations", [])]
    lines += ["", "_References identify retrieved passages, not independent verification of their factual meaning._", ""]
    return "\n".join(lines)


def render_agent_report_markdown(container: Mapping[str, Any], generated: Mapping[str, Any]) -> str:
    """Render the Report Agent's structured sections without altering its claims."""
    r = container.get("report") or container
    lines = [
        "# BlackWing AI — Preliminary Incident Report",
        "",
        "_Every flight leaves a story. Every clue reveals the truth._",
        "",
        "> **AI-assisted draft.** Verify sources and obtain qualified human review before relying on findings.",
        "",
        f"**Case:** `{_safe(r.get('case_id') or container.get('case_id'))}`  ",
        f"**Model:** `{_safe(r.get('model'))}`  ",
        f"**Status:** `{_safe(r.get('status'))}`",
        "",
        "## Executive summary",
        "",
        _safe(generated.get("executive_summary")) or "No summary was produced.",
    ]
    for section in generated.get("sections", []):
        lines += ["", f"## {_safe(section.get('title') or section.get('task_id'))}", ""]
        if section.get("summary"):
            lines += [_safe(section["summary"]), ""]
        for point in section.get("key_points", []):
            lines.append(f"- **{_safe(point.get('classification'))}:** {_safe(point.get('text'))}{_ids(point.get('evidence_ids'))}")
            sources = point.get("source_ids") or []
            if sources:
                lines.append("  - Sources: " + ", ".join(f"`{_safe(source)}`" for source in sources))
        if section.get("generation_method") == "deterministic_fallback":
            lines += ["", "_This section uses a deterministic summary of the source-agent outputs._"]
    lines += [
        "",
        "**Human approval required:** Yes",
        "",
        _safe(generated.get("report_notice") or "Preliminary AI-assisted analysis; human review required."),
        "",
    ]
    return "\n".join(lines)
