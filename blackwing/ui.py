"""Investigation-first Streamlit workstation for BLACKWING AI Copilot."""
import json

import streamlit as st

from blackwing_app import (
    OUTPUT,
    add_case_evidence,
    ask_question,
    assess_risk,
    create_case,
    generate_report,
    get_case,
    list_cases,
    review_finding,
    source_pools,
)
from copilot_report import render_report_pdf
from copilot_state import MAX_NARRATIVE_CHARS, is_protected_source, library, list_historical_cases
from inference import inference_status, provider_name
from safety_register import build_hazard_register
from review_workflow import ROOT

GOA = "GOA-VT-BXL"
PRESET_NAMES = {
    "JFK-DCA23LA125-DEMO": "JFK",
    "AUS-DCA23FA149": "Austin",
    GOA: "Goa",
    "NEW-MUMBAI-SYNTHETIC": "Mumbai (synthetic)",
}


def case_name(incident):
    case_id = incident["case_id"]
    name = incident.get("title") or PRESET_NAMES.get(case_id) or incident.get("airport") or case_id
    details = [name, incident.get("event_date"), incident.get("country") or incident.get("jurisdiction")]
    return " · ".join(str(value) for value in details if value)

def citation_label(citation):
    locator = citation.get("locator")
    page = f" · {locator}" if locator else (f" · p. {citation['page']}" if citation.get("page") else "")
    title = citation.get("title") or citation.get("source_id") or "Evidence"
    scope = citation.get("scope", "")
    return f"{title}{page} · {scope}" if scope else f"{title}{page}"


def finding_label(finding):
    kind = str(finding.get("kind") or finding.get("section") or "finding").replace("_", " ")
    claim = finding["claim"]
    return f"{kind.capitalize()} · {claim[:100]}{'…' if len(claim) > 100 else ''}"


def show_finding(finding, *, excerpts=True):
    st.text(finding["claim"])
    st.caption(
        f"{finding.get('kind', 'finding').replace('_', ' ').capitalize()} · "
        f"{finding.get('status', 'pending').capitalize()}"
    )
    if finding.get("uncertainty"):
        st.text(f"Uncertainty · {finding['uncertainty']}")
    if finding.get("related_finding_ids"):
        st.caption("Related findings · " + ", ".join(finding["related_finding_ids"]))
    if finding.get("reviewed_by"):
        st.text(f"Reviewed by {finding['reviewed_by']} · {finding.get('reviewed_at', '')}")
    if excerpts and finding.get("verification", {}).get("eligible") is False:
        st.warning("The evidence checks flagged this claim. Ask a more focused question or reject it.")
    for citation in finding.get("citations", []):
        if excerpts:
            with st.container(border=True):
                st.caption(citation_label(citation))
                st.code(citation.get("chunk_id") or citation.get("source_id") or "", language=None)
                if citation.get("supporting_quote"):
                    st.caption("Matched supporting quote")
                    st.text(citation["supporting_quote"])
                if citation.get("source_status"):
                    st.caption(str(citation["source_status"]).replace("_", " ").capitalize())
                st.caption("Retrieved passage")
                st.text(citation.get("excerpt") or "Excerpt unavailable.")
        else:
            st.caption(citation_label(citation))


def open_case(case_id):
    st.session_state["workspace_case"] = case_id
    st.session_state["workspace_view"] = "case"
    st.rerun()


def register_case():
    st.header("Start New Investigation")
    with st.form("new-case"):
        airport_field, country_field, date_field = st.columns(3)
        airport = airport_field.text_input("Airport", max_chars=60, placeholder="Name or airport code")
        country = country_field.text_input("Country or jurisdiction", max_chars=60)
        event_date = date_field.date_input("Incident date", value=None)
        aircraft = st.text_input("Aircraft details", max_chars=1000,
                                 placeholder="Flight numbers, aircraft types or registrations, if known")
        narrative = st.text_area("Reported incident", max_chars=MAX_NARRATIVE_CHARS,
                                 placeholder="Describe what was observed and flag uncertain details.")
        uploads = st.file_uploader("Supporting documents", type=["pdf", "txt"],
                                   accept_multiple_files=True, max_upload_size=5)
        provenance = st.text_input("Document provenance", max_chars=500,
                                   placeholder="Source and origin of the submitted documents")
        st.caption("Remove personal identifiers where possible. Uploaded records need corroboration.")
        create = st.form_submit_button("Create investigation", type="primary")
    if create:
        files = [(file.name, file.getvalue()) for file in uploads]
        try:
            incident = create_case(airport, event_date.isoformat() if event_date else None,
                                   country, narrative, aircraft_details=aircraft, files=files,
                                   provenance=provenance.strip() or "User-submitted evidence")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"The investigation could not be created ({type(exc).__name__}).")
        else:
            open_case(incident["case_id"])


def explore_cases(cases, history):
    st.header("Explore Historical Incidents")
    if history:
        choices = {row["case_id"]: case_name(row) for row in history}
        case_id = st.selectbox("Historical incident", list(choices), index=None,
                               format_func=choices.__getitem__, placeholder="Search by airport, date or country")
        if st.button("Open historical incident", type="primary", disabled=case_id is None):
            open_case(case_id)
    else:
        st.info("Historical incidents appear here when eligible case evidence is available.")
    historical_ids = {row["case_id"] for row in history}
    saved = {row["case_id"]: case_name(row) for row in cases
             if row["case_id"] not in historical_ids and not row.get("held_out")}
    if saved:
        st.subheader("Saved investigations")
        case_id = st.selectbox("Investigation", list(saved), index=None, format_func=saved.__getitem__,
                               placeholder="Choose a saved investigation")
        if st.button("Resume investigation", disabled=case_id is None):
            open_case(case_id)
    if any(row.get("held_out") for row in cases):
        st.subheader("Independent evaluation")
        if st.button("Goa factual packet"):
            open_case(GOA)


def accepted_hazards(case):
    return {row["finding_id"]: row for row in case["findings"]
            if row.get("kind") == "hazard" and row.get("status") == "accepted"}


def run_turn(case_id, question, settings, *, mode, comparison_id=None, hazard_id=None, authorized=False):
    if settings["provider"] == "iit" and not authorized:
        st.error("Authorize the IIT transfer for this question before sending.")
        return
    try:
        with st.spinner("Retrieving and checking evidence…"):
            ask_question(case_id, question, mode=mode, comparison_case_id=comparison_id, hazard_id=hazard_id,
                         authorize_hosted=authorized, **settings)
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"The investigation step could not finish ({type(exc).__name__}).")
    else:
        st.rerun()


def authorize_turn(provider):
    if provider != "iit":
        return False
    return st.checkbox("Send this question and the relevant case excerpts to IIT for this turn.",
                       help="The selected gateway receives question context and retrieved evidence.")


def question_form(case_id, settings, mode, label, example, *, comparison_id=None, hazard_id=None):
    with st.form(f"turn:{case_id}:{mode}", clear_on_submit=True):
        question = st.text_area("Question", placeholder=example, max_chars=4000, label_visibility="collapsed")
        authorized = authorize_turn(settings["provider"])
        submitted = st.form_submit_button(label)
    if submitted:
        run_turn(case_id, question.strip() or example, settings, mode=mode, comparison_id=comparison_id,
                 hazard_id=hazard_id, authorized=authorized)


def show_chat(case, history, settings):
    incident = case["incident"]
    case_id = incident["case_id"]
    st.subheader("Investigation conversation")
    if incident.get("held_out"):
        st.info("Goa is reserved for the independent final test. Register its verified factual packet in Evidence and verification.")
    current = {row["finding_id"]: row for row in case["findings"]}
    for turn in case["turns"][-12:]:
        with st.chat_message("user"):
            st.text(turn["question"])
        with st.chat_message("assistant"):
            st.text(turn.get("answer") or "No answer was recorded.")
            status = str(turn.get("status", "")).replace("_", " ").capitalize()
            st.caption(f"{status} · {turn.get('latency_seconds', 0)} s")
            for candidate in turn.get("findings", []):
                with st.container(border=True):
                    show_finding(current.get(candidate["finding_id"], candidate))
    if incident.get("held_out"):
        return
    comparison_id = hazard_id = None
    targets = {row["case_id"]: case_name(row) for row in history if row["case_id"] != case_id}
    hazards = accepted_hazards(case)
    if targets or hazards:
        with st.expander("Context for the next question"):
            if targets:
                comparison_id = st.selectbox("Comparison incident", [None, *targets],
                                              format_func=lambda key: targets.get(key, "Choose from related incidents automatically"),
                                              key=f"chat-compare:{case_id}")
            if hazards:
                hazard_id = st.selectbox("Accepted hazard", [None, *hazards],
                                         format_func=lambda key: finding_label(hazards[key]) if key else "No specific hazard",
                                         key=f"chat-hazard:{case_id}")
    if settings["provider"] == "iit":
        with st.form(f"hosted-chat:{case_id}", clear_on_submit=True):
            question = st.text_area("Ask about this incident", max_chars=4000)
            authorized = authorize_turn("iit")
            sent = st.form_submit_button("Send question", type="primary")
        if sent and question.strip():
            run_turn(case_id, question.strip(), settings, mode="auto", comparison_id=comparison_id,
                     hazard_id=hazard_id, authorized=authorized)
        elif sent:
            st.error("Enter an investigation question.")
    else:
        question = st.chat_input("Ask a question or follow up on this incident", max_chars=4000, key=f"chat:{case_id}")
        if question and question.strip():
            run_turn(case_id, question.strip(), settings, mode="auto", comparison_id=comparison_id, hazard_id=hazard_id)

def show_review(case):
    status = st.radio("Finding status", ["Pending", "Accepted", "Rejected", "All"], horizontal=True)
    matching = {row["finding_id"]: row for row in case["findings"]
                if status == "All" or row.get("status") == status.lower()}
    if not matching:
        st.info(f"No {status.lower() + ' ' if status != 'All' else ''}findings for this case.")
        return

    selected = st.selectbox("Finding", list(matching),
                            format_func=lambda item: finding_label(matching[item]))
    finding = matching[selected]
    show_finding(finding)
    if finding.get("review_note"):
        st.text(f"Review note · {finding['review_note']}")
    if finding.get("status") != "pending":
        return
    st.caption("Acceptance adds the finding to the preliminary report. Consequential decisions require qualified human approval.")
    with st.form(f"review:{selected}"):
        reviewer = st.text_input("Reviewer name", max_chars=100)
        note = st.text_area("Review note", max_chars=500)
        checked = st.checkbox("I checked the cited passages, claim support, and stated uncertainty.")
        accept_col, reject_col = st.columns(2)
        accept = accept_col.form_submit_button(
            "Accept finding", type="primary", disabled=finding.get("verification", {}).get("eligible") is False,
        )
        reject = reject_col.form_submit_button("Reject finding")
    if accept or reject:
        if not reviewer.strip():
            st.error("Enter your reviewer name.")
        elif accept and not checked:
            st.error("Check the evidence and confirm your review before accepting.")
        else:
            try:
                review_finding(case["incident"]["case_id"], selected, "accepted" if accept else "rejected",
                               reviewer.strip(), note.strip())
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.rerun()


def findings_view(case, *, sections=(), kinds=()):
    items = [row for row in case["findings"]
             if row.get("section") in sections or row.get("kind") in kinds]
    if not items:
        st.caption("No findings in this view yet.")
    for item in items:
        with st.container(border=True):
            show_finding(item)


def show_overview(case, settings):
    incident = case["incident"]
    st.subheader("Case overview")
    if incident.get("aircraft_details"):
        st.text(f"Aircraft · {incident['aircraft_details']}")
    if incident.get("narrative"):
        with st.expander("Reported incident", expanded=True):
            st.text(incident["narrative"])
    if not incident.get("held_out"):
        question_form(incident["case_id"], settings, "overview", "Investigate overview",
                      "What does the evidence establish about the airport, aircraft and incident?")
    findings_view(case, sections=("overview",))
    st.subheader("Event timeline")
    if not incident.get("held_out"):
        question_form(incident["case_id"], settings, "chronology", "Investigate timeline",
                      "In what order did the movements, clearances and recovery actions occur?")
    findings_view(case, sections=("chronology",))


def show_comparison(case, history, settings):
    incident = case["incident"]
    case_id = incident["case_id"]
    if not incident.get("held_out"):
        targets = {row["case_id"]: case_name(row) for row in history if row["case_id"] != case_id}
        selected = st.selectbox("Compare with", [None, *targets],
                                format_func=lambda key: targets.get(key, "Select related incidents automatically"),
                                key=f"compare:{case_id}")
        question_form(case_id, settings, "compare", "Investigate similarities and differences",
                      "Which historical incidents are relevant, and how do their hazards and recovery actions compare?",
                      comparison_id=selected)
    findings_view(case, kinds=("comparison",))


def show_regulatory(case, settings):
    incident = case["incident"]
    country = incident.get("country") or incident.get("jurisdiction") or "unspecified"
    st.caption(f"Jurisdiction · {country}  |  Incident date · {incident.get('event_date') or 'unspecified'}")
    st.caption("Candidate guidance requires verification of its jurisdiction, edition and effective date.")
    if not incident.get("held_out"):
        question_form(incident["case_id"], settings, "regulatory", "Investigate regulatory context",
                      "Which candidate procedures should an investigator verify for this jurisdiction and date?")
    findings_view(case, sections=("regulatory",))


def show_hazards(case, register, settings):
    incident = case["incident"]
    case_id = incident["case_id"]
    if not incident.get("held_out"):
        question_form(case_id, settings, "hazards", "Investigate hazards",
                      "What hazards does the case evidence support, and what controls or consequences are documented?")
        hazards = accepted_hazards(case)
        hazard_id = st.selectbox("Risk screen target", [None, *hazards],
                                 format_func=lambda key: finding_label(hazards[key]) if key else "General screen",
                                 key=f"risk:{case_id}")
        if st.button("Run cautious risk screen"):
            try:
                assess_risk(case_id, hazard_id)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.rerun()
    screens = [turn for turn in case["turns"] if turn.get("mode") == "risk"]
    if screens:
        st.text(screens[-1].get("answer", ""))
    if not register:
        st.caption("Cited hazard findings appear here.")
    for entry in register:
        with st.container(border=True):
            show_finding(entry["hazard"])
            st.caption(entry["verification_status"])
            st.text("Potential consequence · " + (entry["potential_consequence"] or "Evidence needed"))
            st.text("Existing controls · " + (", ".join(entry["existing_controls"]) or "Evidence needed"))
            st.text("Uncertainty · " + (entry["uncertainty"] or "Review required"))
            st.markdown("**Risk assessment**")
            if not entry["risk_assessment"]:
                st.caption("No linked risk screen.")
            for finding in entry["risk_assessment"]:
                show_finding(finding, excerpts=False)
            st.markdown("**Proposed mitigations**")
            if not entry["proposed_mitigations"]:
                st.caption("No linked mitigation findings.")
            for finding in entry["proposed_mitigations"]:
                show_finding(finding, excerpts=False)


def show_mitigations(case, register, settings):
    incident = case["incident"]
    if not incident.get("held_out"):
        hazards = accepted_hazards(case)
        if hazards:
            target = st.selectbox("Hazard to address", list(hazards),
                                  format_func=lambda key: finding_label(hazards[key]))
            question_form(incident["case_id"], settings, "mitigations", "Propose a mitigation",
                          "What practical preventive control or review action could address this hazard?",
                          hazard_id=target)
        else:
            st.info("Accept a cited hazard in Evidence and verification before requesting mitigation options.")
    for entry in register:
        if not entry["proposed_mitigations"]:
            continue
        with st.container(border=True):
            st.text(f"Hazard · {entry['hazard']['claim']}")
            for finding in entry["proposed_mitigations"]:
                show_finding(finding)


def show_gaps(case, settings):
    incident = case["incident"]
    if not incident.get("held_out"):
        question_form(incident["case_id"], settings, "gaps", "Check evidence gaps",
                      "Which investigation questions remain unresolved in the supplied records?")
    latest = next((step["results"] for turn in reversed(case["turns"]) for step in turn.get("trace", [])
                   if step.get("step") == "Evidence checklist" and step.get("results")), None)
    if latest:
        st.caption("Coverage of the latest retrieved passages")
        st.dataframe([{"Evidence to check": row["item"].replace("_", " ").capitalize(),
                       "Status": row["status"].replace("_", " ").capitalize()} for row in latest],
                     hide_index=True, width="stretch")
    else:
        st.caption("Run an investigation question to see the evidence coverage checklist.")
    findings_view(case, kinds=("gap", "evidence_gap"))

@st.cache_data
def indexed_sources(case_id, jurisdiction):
    records = [row for row in library() if not is_protected_source(row)]
    direct, guidance, _ = source_pools(records, {"case_id": case_id, "jurisdiction": jurisdiction})
    return list({row["document_id"]: {"Source ID": row["document_id"], "File": row.get("filename"),
                                     "Role": role}
                 for records, role in ((direct, "Case evidence"), (guidance, "Candidate guidance"))
                 for row in records}.values())


def show_evidence(case):
    incident = case["incident"]
    held_out = bool(incident.get("held_out"))
    sources = indexed_sources(incident["case_id"], incident.get("jurisdiction") or "")
    if sources:
        st.caption("Indexed sources")
        st.dataframe(sources, hide_index=True, width="stretch")
    if case["evidence"]:
        st.caption("Submitted files")
        st.dataframe(case["evidence"], hide_index=True, width="stretch",
                     column_order=["filename", "source_role", "source_status", "provenance", "reviewer"])
    if held_out and not any(row.get("source_role") == "evidence" for row in case["evidence"]):
        st.info("Goa is awaiting a separately verified factual evidence packet.")

    source_type = st.selectbox("Source type", ["Factual record", "Candidate regulatory guidance"])
    guidance = source_type == "Candidate regulatory guidance"
    with st.form(f"upload:{incident['case_id']}"):
        files = st.file_uploader("Add supporting evidence", type=["pdf", "txt"],
                                 accept_multiple_files=True, max_upload_size=5)
        provenance = st.text_area("Provenance", max_chars=500,
                                  placeholder="Document origin and how the factual content was obtained.")
        reviewer = ""
        approved = False
        if held_out and not guidance:
            st.caption("Use independent factual material. Keep AAIB final reports and gold answers outside the evidence packet.")
        if held_out or guidance:
            reviewer = st.text_input("Source reviewer", max_chars=100)
            attestation = ("I checked the issuing authority, edition, and jurisdiction of this candidate guidance."
                           if guidance else
                           "I independently checked the packet for factual provenance and held-out material.")
            approved = st.checkbox(attestation)
        add = st.form_submit_button("Save evidence")
    if add:
        if not files or not provenance.strip():
            st.error("Choose at least one supporting file and describe its provenance.")
        elif (held_out or guidance) and (not approved or not reviewer.strip()):
            st.error("Enter the source reviewer and confirm the review.")
        else:
            try:
                add_case_evidence(incident["case_id"], [(file.name, file.getvalue()) for file in files],
                                  provenance=provenance.strip(), reviewer=reviewer.strip(), approved=approved,
                                  source_role="rule_candidate" if guidance else "evidence")
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"The evidence could not be saved ({type(exc).__name__}).")
            else:
                st.rerun()


def show_report(case):
    report = generate_report(case["incident"]["case_id"])
    count = report["accepted_count"]
    if not count:
        st.info("Accept a cited finding in Review findings to enable PDF export.")
    else:
        st.download_button(
            "Download preliminary report", render_report_pdf(report),
            file_name=f"BLACKWING-{case['incident']['case_id']}-preliminary.pdf",
            mime="application/pdf", type="primary",
        )
        st.caption(f"{count} accepted {'finding' if count == 1 else 'findings'}")
    for section in report["sections"]:
        with st.expander(section["title"]):
            if not section["findings"]:
                st.caption("No reviewed findings for this section.")
            for finding in section["findings"]:
                st.text(finding["claim"])
                references = finding.get("citation_refs", [])
                if references:
                    st.caption("Sources · " + ", ".join(str(reference) for reference in references))
    if report.get("sources"):
        with st.expander("Report sources"):
            for source in report["sources"]:
                st.caption(f"{source['number']} · {citation_label(source)}")
                st.text(source["excerpt"])


def show_history(case, *, include_legacy=True):
    turns = list(reversed(case["turns"]))[:25]
    if turns:
        labels = {turn["turn_id"]: f"{turn.get('mode', 'Investigation')} · {turn.get('question', '')[:90]}"
                  for turn in turns}
        selected = st.selectbox("Inspect a Copilot trace", list(labels), format_func=labels.__getitem__)
        turn = next(row for row in turns if row["turn_id"] == selected)
        st.json(turn["trace"], expanded=False)
    else:
        st.caption("Completed and failed investigation turns appear here.")

    if not include_legacy:
        return
    earlier = {}
    for path in sorted(OUTPUT.glob("brief-*.json"), key=lambda item: item.name):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if record.get("case_id") == case["incident"]["case_id"] and record.get("run_id"):
            earlier[record["run_id"]] = record
    if earlier:
        with st.expander("Earlier full-brief traces"):
            selected = st.selectbox("Inspect an earlier execution", list(earlier))
            st.json(earlier[selected].get("trace", []), expanded=False)


def show_inference(*, cloud=False):
    providers = {"local": "Local Ollama", "iit": "IIT Hosted"}
    if cloud:
        provider = "iit"
    else:
        provider = st.selectbox("Inference provider", list(providers), index=list(providers).index(provider_name()),
                                format_func=providers.__getitem__, key="inference-provider")
    status = inference_status(provider)
    st.caption(f"{status.get('provider_label') or providers[provider]} · {status.get('model') or 'Model unavailable'}")
    st.text(f"API status · {status.get('api_status', 'Not checked')}")
    tokens = status.get("tokens_used")
    st.text(f"Tokens used · {tokens if tokens is not None else 'Unavailable'}")
    st.text(f"Requests completed · {status.get('requests_completed', 0)} of {status.get('requests_attempted', 0)}")
    if status.get("remaining_quota") is not None:
        st.text(f"Remaining quota · {status['remaining_quota']}")
    if cloud:
        retrieval = "BM25"
        st.caption("Retrieval · BM25")
        if status["api_status"] in {"Not configured", "Disabled"}:
            st.info("IIT inference is disabled or awaiting configuration. Evidence browsing and the risk screen are available.")
    else:
        retrieval = st.selectbox("Retrieval", ["Hybrid + reranker", "BM25"])
    cap = st.selectbox("Output token cap", [512, 1024])
    settings = {"provider": provider, "retrieval": retrieval, "cap": cap}
    if cloud:
        settings["cache"] = False
    return settings


def main(*, cloud=False):
    st.set_page_config(page_title="BLACKWING AI Copilot", layout="wide", page_icon="✈")
    st.title("BLACKWING AI Copilot")
    if cloud:
        st.caption("Cloud demonstration · Investigation state is temporary and scoped to this browser session.")
    with st.sidebar:
        st.header("Case workspace")
        if st.button("Start New Investigation", width="stretch"):
            st.session_state["workspace_view"] = "new"
        if st.button("Explore Historical Incidents", width="stretch"):
            st.session_state["workspace_view"] = "history"
        with st.expander("Inference status", expanded=True):
            settings = show_inference(cloud=cloud)
        glossary = ROOT / "aviation-glossary.pdf"
        if glossary.is_file():
            st.download_button("Aviation glossary", glossary.read_bytes(),
                               file_name="BLACKWING-Aviation-Glossary.pdf", mime="application/pdf")

    view = st.session_state.get("workspace_view", "home")
    if view == "home":
        st.markdown("Investigate an aviation safety incident and review findings against the supporting evidence.")
        new, historical = st.columns(2)
        with new:
            with st.container(border=True):
                st.subheader("New investigation")
                st.write("Enter an incident narrative and attach supporting records.")
                if st.button("Start New Investigation", type="primary", key="home-new", width="stretch"):
                    st.session_state["workspace_view"] = "new"
                    st.rerun()
        with historical:
            with st.container(border=True):
                st.subheader("Historical incidents")
                st.write("Browse the case library and reopen saved investigations.")
                if st.button("Explore Historical Incidents", key="home-history", width="stretch"):
                    st.session_state["workspace_view"] = "history"
                    st.rerun()
        return
    if view == "new":
        register_case()
        return
    history = list_historical_cases()
    if view == "history":
        cases = [case for case in list_cases() if not (cloud and case.get('held_out'))]
        explore_cases(cases, history)
        return

    case = get_case(st.session_state["workspace_case"])
    incident = case["incident"]
    st.header(case_name(incident))
    st.caption(f"Case ID · {incident['case_id']}")
    if incident.get("held_out"):
        st.info("Independent evaluation case. Interactive model calls are held for the frozen Goa evaluation.")
    register = build_hazard_register(case["findings"])
    labels = ("Q&A", "Overview and timeline", "Historical comparison", "Regulatory context",
              "Hazards and risk", "Mitigation plan", "Evidence and verification", "Report", "Run history")
    chat, overview, comparison, regulatory, hazards, mitigations, evidence, report, history_tab = st.tabs(labels)
    with chat:
        show_chat(case, history, settings)
    with overview:
        show_overview(case, settings)
    with comparison:
        show_comparison(case, history, settings)
    with regulatory:
        show_regulatory(case, settings)
    with hazards:
        show_hazards(case, register, settings)
    with mitigations:
        show_mitigations(case, register, settings)
    with evidence:
        st.subheader("Evidence gaps")
        show_gaps(case, settings)
        st.subheader("Sources")
        show_evidence(case)
        st.subheader("Review findings")
        show_review(case)
    with report:
        show_report(case)
    with history_tab:
        show_history(case, include_legacy=not cloud)


if __name__ == "__main__":
    main()
