from __future__ import annotations

from io import BytesIO
import time
from html import escape
from typing import Any

import httpx
import streamlit as st
from blackwing.config import Settings
from blackwing.rag.datasets import load_demo_inputs
from blackwing.reporting import render_markdown_report, safe_report_text
from blackwing.agents.risk.scoring import LIKELIHOOD_LEVELS, SEVERITY_LEVELS


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


golden_cases = load_demo_inputs("golden")
golden_case = golden_cases[0] if golden_cases else None
use_golden_demo = st.checkbox(
    "Load golden demo input (held out from the RAG index)",
    help="Prefills the demo request. Attach its preliminary report below as current-case evidence.",
    disabled=golden_case is None,
)

with st.form("incident_form"):
    st.markdown('<div class="bw-kicker">Start an investigation</div>', unsafe_allow_html=True)
    narrative = st.text_area(
        "Incident narrative",
        value=golden_case["narrative"] if use_golden_demo and golden_case else "",
        key=f"incident_narrative_{'golden' if use_golden_demo else 'custom'}",
        height=210,
        placeholder="Describe the event and include any known date, airport, aircraft, and source references.",
    )
    uploaded_report = st.file_uploader(
        "Attach source report (optional)",
        type=["pdf", "txt"],
        help="Uploaded text is attached to this case as unverified evidence; it is not added to the shared RAG index.",
    )
    submitted = st.form_submit_button("Analyze incident", type="primary")


def _uploaded_evidence(file: Any) -> list[dict]:
    if file is None:
        return []
    filename = file.name
    if filename.lower().endswith(".txt"):
        pages = [file.getvalue().decode("utf-8-sig", errors="replace")]
    else:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("PDF attachment support requires `pip install -r requirements-rag.txt`.") from exc
        try:
            reader = PdfReader(BytesIO(file.getvalue()))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("The attached PDF is password protected and could not be read.")
            pages = [(page.extract_text() or "") for page in reader.pages]
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"Could not extract text from the attached PDF ({type(exc).__name__}).") from exc
    evidence = []
    for page_number, content in enumerate(pages, 1):
        content = content.strip()
        if not content:
            continue
        # Split extracted pages for transport while preserving every character.
        for offset in range(0, len(content), 19_500):
            excerpt = content[offset:offset + 19_500]
            evidence.append({
                "title": filename[:300],
                "document_type": "incident_report",
                "source_status": "uploaded_unverified",
                "page_number": page_number,
                "section_reference": f"text segment {offset // 19_500 + 1}" if len(content) > 19_500 else None,
                "excerpt": excerpt,
            })
    return evidence


def _render_report(generated: dict, report: dict, case_id: str, run_id: str) -> None:
    st.markdown("## Preliminary incident report")
    st.caption(f"Case {case_id}  ·  Run {run_id}  ·  {report.get('model', 'Configured model')}")
    st.markdown(
        f"<div class='bw-summary'><div class='bw-kicker'>Executive summary</div>"
        f"<p>{escape(safe_report_text(generated.get('executive_summary') or 'No executive summary was produced.'))}</p></div>",
        unsafe_allow_html=True,
    )

    agent_outputs = report.get("agent_outputs", {})
    similar = agent_outputs.get("similar_case", {})
    risk_output = agent_outputs.get("risk") or {}
    risk_score = risk_output.get("numeric_score")
    left, middle, right, review = st.columns(4)
    left.metric("Case match", "Unseen" if str(similar.get("case_status", "")).startswith("unseen_case") else "Compared")
    score_display = (
        f"{risk_score}/25 · {risk_output.get('risk_rating')}"
        if risk_score is not None else "Not scored"
    )
    score_label = "Risk score (partial)" if "partial" in str(risk_output.get("assessment_status", "")) else "Risk score (provisional)"
    middle.metric(score_label, score_display)
    right.metric("Review", "Required" if generated.get("human_approval_required", True) else "Pending")
    review.metric(
        "Evidence check",
        "Needs verification" if (agent_outputs.get("verification") or {}).get("completeness_status") == "incomplete" else "Reviewed",
    )
    st.caption(
        "Provisional 5×5 score = severity × likelihood; overall score is the highest scored hazard. "
        "Severity: 1 Negligible–5 Catastrophic; likelihood: 1 Rare–5 Almost certain. "
        "Bands: 1–4 Low, 5–9 Moderate, 10–16 High, 17–25 Extreme. Human review required."
    )
    if risk_output.get("hazards"):
        risk_rows = []
        for hazard in risk_output["hazards"]:
            severity = hazard.get("severity")
            likelihood = hazard.get("likelihood")
            risk_rows.append({
                "Hazard": hazard.get("hazard_id", ""),
                "Severity": f"{severity} · {SEVERITY_LEVELS.get(severity, 'Unrated')}" if severity else "Unrated",
                "Likelihood": f"{likelihood} · {LIKELIHOOD_LEVELS.get(likelihood, 'Unrated')}" if likelihood else "Unrated",
                "Score": f"{hazard.get('score')}/25 · {hazard.get('risk_rating')}" if hazard.get("score") is not None else "Not scored",
                "Hazard": hazard.get("description", ""),
                "Severity basis": hazard.get("severity_rationale", ""),
                "Likelihood basis": hazard.get("likelihood_rationale", ""),
            })
        st.dataframe(risk_rows, hide_index=True, use_container_width=True)
    for missing in risk_output.get("missing_inputs", []):
        st.caption("Risk scoring gap: " + safe_report_text(missing))

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


if submitted:
    if use_golden_demo and uploaded_report is None:
        st.error("Attach the golden case's preliminary report as evidence before starting this demo.")
    elif len(narrative.strip()) < 20:
        st.error("Enter at least 20 characters of incident narrative.")
    else:
        settings = None
        try:
            uploaded_evidence = _uploaded_evidence(uploaded_report)
            if use_golden_demo and not uploaded_evidence:
                raise ValueError("The attached report has no extractable text. Scanned PDFs need OCR before this demo can use them.")
            settings = Settings.load()
            with httpx.Client(timeout=settings.ui_request_timeout_seconds) as client:
                case_response = client.post(
                    f"{settings.ui_api_url}/api/v1/cases",
                    json={
                        "category": "runway_incursion",
                        "narrative": narrative,
                        "evidence": uploaded_evidence,
                        **({
                            "source_incident_id": golden_case["case_id"],
                            "incident_date": golden_case["incident_date"],
                            "jurisdiction": golden_case["jurisdiction"],
                            "location": golden_case["case_name"],
                        } if use_golden_demo and golden_case else {}),
                    },
                )
                case_response.raise_for_status()
                case = case_response.json()
                with st.spinner("Agents are reviewing evidence and drafting the report in focused sections…"):
                    run_response = client.post(f"{settings.ui_api_url}/api/v1/cases/{case['case_id']}/runs", json={})
                    run_response.raise_for_status()
                    run = run_response.json()
                    deadline = time.monotonic() + settings.ui_request_timeout_seconds
                    poll_errors = 0
                    while run["status"] in {"queued", "running"} and time.monotonic() < deadline:
                        time.sleep(2)
                        try:
                            status_response = client.get(f"{settings.ui_api_url}/api/v1/runs/{run['run_id']}")
                            status_response.raise_for_status()
                            run = status_response.json()
                            poll_errors = 0
                        except httpx.RequestError:
                            poll_errors += 1
                            st.warning(f"The API connection dropped while checking the run (attempt {poll_errors}); reconnecting…")
                            time.sleep(min(2 * poll_errors, 10))

            st.markdown("## Investigation run")
            st.caption(f"Case ID: {case['case_id']}  ·  Run ID: {run['run_id']}  ·  Status: {run['status']}")
            if run["status"] == "completed":
                response = httpx.get(
                    f"{settings.ui_api_url}/api/v1/cases/{case['case_id']}/report",
                    timeout=settings.ui_request_timeout_seconds,
                )
                response.raise_for_status()
                result = response.json()
                report = result["report"]
                generated = report.get("generated_report")
                if generated:
                    report_tab, agents_tab, trace_tab = st.tabs(["Investigation report", "Agent details", "Run trace"])
                    with report_tab:
                        _render_report(generated, report, case["case_id"], run["run_id"])
                    with agents_tab:
                        st.markdown("### Source agent work products")
                        st.caption("Structured JSON stays available for review and citation checking.")
                        for name, output in report.get("agent_outputs", {}).items():
                            with st.expander(name.replace("_", " ").title(), expanded=False):
                                if name == "similar_case" and str(output.get("case_status", "")).startswith("unseen_case"):
                                    st.info("Retrieved reports are historical analogies, not findings about this incident.")
                                st.json(output)
                    with trace_tab:
                        trace_response = httpx.get(
                            f"{settings.ui_api_url}/api/v1/runs/{run['run_id']}/trace",
                            timeout=settings.ui_request_timeout_seconds,
                        )
                        trace_response.raise_for_status()
                        st.json({"trace": trace_response.json(), "report_tasks": generated.get("sections", [])})
                        with st.expander("Limitations and review boundary", expanded=True):
                            st.write(report.get("limitations", []))
                            st.info(result.get("notice", "Human review required."))
                else:
                    # Backward compatibility for reports created before the Report Agent.
                    st.markdown(render_markdown_report(result))
                    for name, output in report.get("agent_outputs", {}).items():
                        with st.expander(name.replace("_", " ").title(), expanded=False):
                            st.json(output)
            elif run["status"] == "failed":
                st.error(run.get("error") or "The analysis run failed.")
            else:
                st.info(f"Run is still {run['status']}. Poll GET /api/v1/runs/{run['run_id']} for status.")
        except httpx.ConnectError as exc:
            api_url = settings.ui_api_url if settings else "configured API URL"
            st.error(f"Could not connect to the BlackWing API at {api_url}. Check that its server is running and that the UI API URL matches its host and port. ({exc})")
        except httpx.TimeoutException:
            st.error("The BlackWing API request timed out. Check the API run status and hosted model response time.")
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", "")
            except ValueError:
                detail = ""
            st.error(f"The BlackWing API returned HTTP {exc.response.status_code}. {detail}")
        except httpx.HTTPError as exc:
            st.error(f"The BlackWing API request failed: {exc}")
        except (ValueError, OSError) as exc:
            st.error(f"Could not load application settings or case evidence: {exc}")
