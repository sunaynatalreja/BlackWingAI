"""Integrated BLACKWING workstation; existing brand/report rendering retained."""
from pathlib import Path
import sys
from html import escape

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)

import streamlit as st
from blackwing.reporting import safe_report_text
from blackwing.copilot_state import risk_display


def render_brand():
    st.markdown(
        """
        <style>
          :root { --bw-ink:#102337; --bw-muted:#627487; --bw-blue:#12395a; --bw-cyan:#66d5d0; --bw-orange:#ef9d65; }
          .stApp { background:linear-gradient(180deg,#eaf1f6 0%,#f6f8fa 46%,#edf3f7 100%); color:var(--bw-ink); }
          [data-testid="stHeader"] { background:transparent; }
          .block-container { max-width:1440px; padding-top:1.4rem; padding-bottom:4rem; }
          .bw-hero { position:relative; overflow:hidden; display:flex; align-items:center; justify-content:space-between;
            min-height:220px; padding:2.2rem 3rem; border-radius:26px; color:#f5fbff;
            background:radial-gradient(circle at 78% 48%,rgba(70,160,190,.20),transparent 23%),
            linear-gradient(122deg,#091a2a 0%,#0d2c47 58%,#123d5a 100%); box-shadow:0 20px 48px rgba(12,42,65,.18); }
          .bw-hero-copy { position:relative; z-index:2; }
          .bw-eyebrow { color:#9adbdc; font-size:.72rem; letter-spacing:.19em; font-weight:700; margin-bottom:.55rem; }
          .bw-title { color:#fff; font-size:clamp(2.2rem,4vw,3.2rem); line-height:1.04; font-weight:760; letter-spacing:-.045em; margin:0; }
          .bw-tagline { color:#d6e8f1; margin:.75rem 0 0; font-size:1.12rem; letter-spacing:.005em; }
          .bw-subline { color:#9bb5c6; margin:.9rem 0 0; font-size:.9rem; }
          .bw-art { width:min(33%,330px); min-width:190px; opacity:.94; filter:drop-shadow(0 0 20px rgba(91,220,220,.18)); }
          .bw-art svg { width:100%; height:auto; }
          .bw-art-ring { fill:none; stroke:rgba(114,218,219,.32); stroke-width:1.2; }
          .bw-art-track { fill:none; stroke:#8ad9d8; stroke-width:2; stroke-dasharray:5 6; }
          .bw-art-runway { fill:rgba(255,255,255,.10); stroke:rgba(255,255,255,.65); stroke-width:1.5; }
          .bw-art-light { fill:#f0a76d; }
          .bw-warning { border:1px solid #ecd9b7; border-left:4px solid #df9c57; border-radius:12px;
            background:#fff9ed; padding:.85rem 1rem; color:#735328; margin:1rem 0 1.3rem; }
          .bw-kicker { color:#47738f; text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:750; }
          .bw-summary { background:linear-gradient(120deg,#fff 0%,#f5fbfd 100%); border:1px solid #d8e5eb;
            border-radius:18px; padding:1.25rem 1.45rem; box-shadow:0 8px 22px rgba(21,54,76,.05); }
          .bw-summary p { font-size:1.07rem; line-height:1.7; margin:.45rem 0 .15rem; color:#18354d; }
          .bw-chip { display:inline-block; border-radius:999px; padding:.22rem .62rem; margin:.08rem .25rem .08rem 0;
            font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; background:#e6f2f4; color:#1d686d; }
          .bw-section { margin:1rem 0; padding:1.15rem 1.35rem; border:1px solid #d8e3e9; border-radius:16px;
            background:rgba(255,255,255,.9); box-shadow:0 7px 22px rgba(21,54,76,.045); }
          .bw-section h3 { margin:.1rem 0 .55rem; color:#163750; }
          .bw-section-summary { color:#2e485c; line-height:1.7; }
          .bw-refs { color:#687f90; font-size:.78rem; margin:.2rem 0 .5rem; }
          .bw-fallback { color:#738796; font-size:.74rem; font-style:italic; margin:.7rem 0 0; }
          [data-testid="stVerticalBlockBorderWrapper"] { border-color:#d8e3e9 !important; border-radius:16px !important; background:rgba(255,255,255,.86); }
          [data-testid="stForm"] { border:1px solid #d8e3e9; border-radius:16px; padding:1.2rem 1.3rem 1rem; background:rgba(255,255,255,.88); }
          div.stButton > button, div.stFormSubmitButton > button { border-radius:10px; border:0; font-weight:700; padding:.62rem 1.05rem;
            background:linear-gradient(110deg,#0d5373,#187c88); color:#fff; box-shadow:0 7px 18px rgba(18,88,112,.18); }
          div.stButton > button:hover, div.stFormSubmitButton > button:hover { background:linear-gradient(110deg,#0b405f,#126c7b); color:#fff; border:0; }
          [data-testid="stTabs"] button { font-weight:700; }
          h2,h3 { color:#163750; letter-spacing:-.02em; }
          .bw-point { margin:.55rem 0; line-height:1.62; }
          .bw-point-label { color:#287279; font-size:.69rem; letter-spacing:.08em; font-weight:800; text-transform:uppercase; }
          .bw-footer { color:#6e8494; font-size:.78rem; padding-top:1.2rem; }
          @media (max-width:700px) { .bw-hero { padding:1.6rem; min-height:190px; } .bw-art { width:30%; min-width:105px; } .bw-tagline { font-size:.98rem; } }
        </style>
        <section class="bw-hero">
          <div class="bw-hero-copy">
            <div class="bw-eyebrow">FLIGHT SAFETY INTELLIGENCE</div>
            <h1 class="bw-title">BlackWing AI</h1>
            <p class="bw-tagline">Every flight leaves a story. Every clue reveals the truth.</p>
            <p class="bw-subline">Evidence-led incident analysis · Human-reviewed decision support</p>
          </div>
          <div class="bw-art" aria-hidden="true">
            <svg viewBox="0 0 300 190" role="img">
              <circle class="bw-art-ring" cx="177" cy="94" r="70"/><circle class="bw-art-ring" cx="177" cy="94" r="48"/>
              <circle class="bw-art-ring" cx="177" cy="94" r="25"/>
              <path class="bw-art-track" d="M34 137 C91 136 108 119 132 101 S187 71 250 51"/>
              <path class="bw-art-runway" d="M203 120 L270 48 L281 58 L214 130 Z"/>
              <path d="M201 116 l11 4 -6 6 -3-1 -4 4 -2-2 3-5z" fill="#a4e4df"/>
              <circle class="bw-art-light" cx="118" cy="111" r="4"/><circle class="bw-art-light" cx="250" cy="62" r="4"/>
              <path d="M20 153h86" stroke="rgba(255,255,255,.35)" stroke-width="1"/>
              <text x="22" y="169" fill="#a9c7d7" font-size="9" letter-spacing="2">FOLLOW THE EVIDENCE</text>
            </svg>
          </div>
        </section>
        <div class="bw-warning"><strong>Preliminary analysis.</strong> Model output is not a factual determination or operational directive. Verify source material and obtain qualified human review.</div>
        """,
        unsafe_allow_html=True,
    )


def _render_report(generated: dict, report: dict, case_id: str, run_id: str, *, reference_case=None) -> None:
    st.markdown("## Preliminary incident report")
    st.caption(f"Case {case_id}  ·  Run {run_id}  ·  {report.get('model', 'Configured model')}")
    st.markdown(
        f"<div class='bw-summary'><div class='bw-kicker'>Executive summary</div>"
        f"<p>{escape(safe_report_text(generated.get('executive_summary') or 'No executive summary was produced.'))}</p></div>",
        unsafe_allow_html=True,
    )

    agent_outputs = report.get("agent_outputs", {})
    similar = agent_outputs.get("similar_case", {})
    score_status = (agent_outputs.get("risk") or {}).get("assessment_status", "provisional")
    left, middle, right, review = st.columns(4)
    display = risk_display(report)
    indices = list(dict.fromkeys(row["Risk index"] for row in display["rows"] if row["Risk index"] != "Unassessed"))
    left.metric("Case match", "Seen" if reference_case is True else "No reference match" if reference_case is False else "Unknown")
    middle.metric("Aviation risk", " · ".join(indices) if indices else "Unassessed")
    right.metric("Review", "Required" if generated.get("human_approval_required", True) else "Pending")
    review.metric(
        "Evidence check",
        "Needs verification" if (agent_outputs.get("verification") or {}).get("completeness_status") == "incomplete" else "Human verification required",
    )

    st.caption("Case match identifies a preserved reference case, not model training exposure. Historical analogies are displayed separately.")
    st.caption(display["framework"])
    st.caption(display["notice"])
    if display["rows"]:
        st.dataframe([{key:row[key] for key in ("Hazard","Severity","Likelihood","Risk index","Classification","Status")}
                      for row in display["rows"]], hide_index=True, alt="Backend hazard severity, likelihood and categorical risk indexes")
    if not indices:
        st.info("This saved backend assessment has no evidence-backed severity and likelihood pair, so no A1/C3-style risk index was assigned. The UI does not invent ratings.")
    for missing in display["missing_inputs"]:
        st.caption("Risk assessment gap: " + str(missing))

    fallback_sections = [s for s in generated.get("sections", []) if s.get("generation_method") == "deterministic_fallback"]
    if fallback_sections:
        st.info("Report sections were assembled from validated agent outputs by the bounded backend workflow. See the task trace for details.")

    for section in generated.get("sections", []):
        title = section.get('title') or section.get('task_id', 'Report section').replace('_', ' ').title()
        html_parts = ["<article class='bw-section'>", f"<h3>{escape(title)}</h3>"]
        if section.get("summary"):
            html_parts.append(f"<p class='bw-section-summary'>{escape(safe_report_text(section['summary']))}</p>")
        for point in section.get("key_points", []):
            classification = escape(point.get("classification", "verification_gap").replace("_", " "))
            html_parts.append(
                f"<div class='bw-point'><span class='bw-point-label'>{classification}</span><br>"
                f"{escape(safe_report_text(point.get('text', '')))}</div>"
            )
            refs = [*point.get("source_ids", []), *point.get("evidence_ids", [])]
            if refs:
                html_parts.append(f"<div class='bw-refs'>Sources: {escape(safe_report_text(' · '.join(str(item) for item in refs)))}</div>")
        if section.get("generation_method") == "deterministic_fallback":
            html_parts.append("<p class='bw-fallback'>Deterministic summary from agent work products</p>")
        html_parts.append("</article>")
        st.markdown("".join(html_parts), unsafe_allow_html=True)

    st.markdown("<div class='bw-footer'>This report is preliminary. Human verification and approval are required.</div>", unsafe_allow_html=True)

def main(*, cloud=False):
    from blackwing.copilot_ui import main as workspace
    workspace(brand=render_brand, render_report=_render_report, cloud=cloud)


if __name__ == "__main__":
    main()
