"""Sunayana workstation layout integrated with the canonical FastAPI backend."""
import json

import httpx
import streamlit as st

from blackwing import copilot
from blackwing.copilot_report import render_report_pdf
from blackwing.copilot_state import MAX_NARRATIVE_CHARS, case_name, risk_display, report_availability, execution_label
from blackwing.report_quality import assess_quality, build_evidence_checklist


def open_case(case_id):
    st.session_state["workspace_case"] = case_id
    st.session_state["workspace_view"] = "case"
    st.rerun()


def quality_panel(incident, report=None):
    # Prompt Completeness measures supplied intake only. Report Completeness
    # never imports that score; aviation risk remains in its existing own view.
    prompt_panel, report_panel = st.columns(2)
    for panel, quality in ((prompt_panel, assess_quality(incident)),
                           (report_panel, assess_quality(incident, report) if report is not None else None)):
        with panel:
            if quality is None:
                st.subheader("Report Completeness")
                st.metric("Report Completeness", "Not available")
                st.caption("A completed backend report is required; no report score has been assigned.")
                continue
            st.subheader(quality["label"])
            st.metric(quality["label"], f"{quality['score']}%")
            st.caption(quality["method"])
            if quality["flags"]:
                st.caption(f"Weighted total before the disclosed integrity cap: {quality['raw_score']}/100.")
                for flag in quality["flags"]:
                    st.warning(flag)
            st.info(quality["notice"])
            st.dataframe(quality["checks"], hide_index=True, alt=quality["label"] + " weights, earned points and missing-information actions")
            if quality["missing_information"]:
                st.write(quality["missing_information"])


def register_case():
    st.header("Start New Investigation")
    with st.form("new-case"):
        airport, country, when = st.columns(3)
        location = airport.text_input("Airport", max_chars=300, key="intake-location")
        jurisdiction = country.text_input("Country or jurisdiction", max_chars=100, key="intake-jurisdiction")
        date = when.date_input("Incident date", value=None, key="intake-date")
        source_id = st.text_input("External incident identifier (if known)", max_chars=100, key="intake-source")
        narrative = st.text_area("Reported incident", max_chars=MAX_NARRATIVE_CHARS, height=210, key="intake-narrative")
        files = st.file_uploader("Supporting documents", type=["pdf", "txt"], accept_multiple_files=True, max_upload_size=5, key="intake-files")
        provenance = st.text_input("Document provenance", max_chars=200, key="intake-provenance")
        st.caption("PDF text is extracted locally. Scanned PDFs need a transcription. No model runs on intake.")
        create = st.form_submit_button("Create investigation", type="primary", key="create-investigation")
        assess = st.form_submit_button("Check missing information", key="check-intake")
    draft = {"narrative": narrative, "location": location, "jurisdiction": jurisdiction,
             "incident_date": date.isoformat() if date else None, "source_incident_id": source_id}
    if assess:
        quality_panel({**draft, "evidence": [{"title": f.name, "source_id": provenance} for f in files] if provenance.strip() else []})
    if create:
        if len(narrative.strip()) < 20:
            st.error("Enter at least 20 characters of incident narrative.")
            return
        evidence = copilot.file_evidence([(f.name, f.getvalue()) for f in files], provenance)
        incident = copilot.create_case(**draft, evidence=evidence)
        open_case(incident["case_id"])


def explore_cases():
    st.header("Explore Historical Incidents")
    st.caption("Saved investigations are read from the backend SQLite repository. Indexed analogies remain in the Historical comparison tab.")
    cases = copilot.list_cases()
    if not cases:
        st.info("No backend investigations have been saved yet.")
        return
    choices = {row["case_id"]: case_name(row) for row in cases}
    selected = st.selectbox("Investigation", list(choices), format_func=choices.__getitem__, key="saved-case")
    if st.button("Resume investigation", key="resume-case"):
        open_case(selected)


def show_finding(finding):
    st.text(finding["claim"])
    st.caption(f"{finding['kind']} · {finding['status']}")
    if finding.get("reviewed_by"):
        st.caption(f"Reviewed by {finding['reviewed_by']} · {finding['reviewed_at']}")
    for citation in finding["citations"]:
        st.caption(f"Source · {citation['source_id']} · {citation['chunk_id']}")
        st.text(citation["excerpt"] or "Excerpt unavailable through the report API; verify the original source.")


def show_review(case):
    st.subheader("Review findings")
    st.caption("Decisions are stored by the backend and bound to the displayed run. Report output remains preliminary.")
    rows = {r["finding_id"]: r for r in case["findings"]}
    if not rows:
        st.info("No backend findings are ready for review.")
        return
    selected = st.selectbox("Finding", list(rows), format_func=lambda key: rows[key]["claim"][:100], key=f"review-finding:{case['report_run_id']}")
    show_finding(rows[selected])
    with st.form(f"review:{case['report_run_id']}:{selected}"):
        reviewer = st.text_input("Reviewer name", max_chars=200, key="reviewer-name")
        rationale = st.text_area("Review rationale", max_chars=4000, key="review-rationale")
        checked = st.checkbox("I checked the original sources, claim support and uncertainty.", key="review-confirmed")
        accept = st.form_submit_button("Accept finding", key="accept-finding")
        reject = st.form_submit_button("Reject finding", key="reject-finding")
    if accept or reject:
        if not reviewer.strip() or not rationale.strip() or (accept and not checked):
            st.error("Supply a reviewer and rationale, and confirm source review before accepting.")
        else:
            copilot.review_finding(case, selected, "accept" if accept else "reject", reviewer.strip(), rationale.strip())
            st.rerun()


def show_evidence(case, *, include_quality=True):
    st.subheader("Evidence gaps")
    incident = case["incident"]
    checklist = build_evidence_checklist({**incident, "event_date": incident.get("incident_date")},
                                        [{"text": e["excerpt"]} for e in case["evidence"]])
    st.caption("Preserved source-branch checklist: topic mentions in supplied text, not verified facts or a requirement to obtain protected final reports.")
    st.dataframe(checklist, hide_index=True, alt="Evidence topic coverage checklist")
    if include_quality:
        quality_panel(incident, case["report"])
    st.subheader("Submitted sources")
    for item in case["evidence"]:
        with st.expander(item["title"]):
            st.caption(f"{item.get('source_id') or item['evidence_id']} · {item.get('source_status') or 'user submitted'}")
            st.text(item["excerpt"])
    if incident["status"] == "draft":
        with st.form("add-evidence"):
            files = st.file_uploader("Add supporting evidence", type=["pdf", "txt"], accept_multiple_files=True, max_upload_size=5, key="evidence-files")
            provenance = st.text_input("Provenance", max_chars=200, key="evidence-provenance")
            add = st.form_submit_button("Save evidence", key="save-evidence")
        if add:
            evidence = copilot.file_evidence([(f.name, f.getvalue()) for f in files], provenance)
            if not evidence:
                st.error("Choose supporting files.")
                return
            for item in evidence:
                copilot.add_case_evidence(incident["case_id"], item)
            st.rerun()
    else:
        st.info("Evidence is locked after analysis starts. Create a new case revision to change its evidence.")
    show_review(case)


def show_risk(report):
    display = risk_display(report)
    st.subheader("Aviation risk assessment")
    st.caption(display["framework"])
    st.warning(display["notice"])
    st.text("Backend assessment status · " + display["assessment_status"])
    if display["rows"]:
        st.dataframe(display["rows"], hide_index=True, alt="Backend-calculated severity, likelihood and categorical risk")
    else:
        st.info("Unassessed: no backend hazard risk values are available.")
    if display["missing_inputs"]:
        st.write(display["missing_inputs"])
    for h in (report or {}).get("agent_outputs", {}).get("risk", {}).get("hazards", []):
        with st.expander(f"Risk basis · {h['hazard_id']}"):
            st.text("Severity basis · " + (h.get("severity_basis") or "Missing"))
            st.text("Likelihood basis · " + (h.get("likelihood_basis") or "Missing"))
            st.text("Existing controls · " + ", ".join(h.get("existing_controls") or []))
            st.text("Control gaps · " + ", ".join(h.get("control_gaps") or []))


def show_report(case, render_report):
    report = case["report"]
    if not report:
        availability = case.get("report_availability") or report_availability(case["runs"], None)
        if availability["state"] in {"failed", "interrupted", "unavailable"}:
            st.warning(availability["message"])
        else:
            st.info(availability["message"])
        return
    reference_case = show_gold_comparison(case)
    if report.get("generated_report"):
        render_report(report["generated_report"], report, case["incident"]["case_id"], case["report_run_id"], reference_case=reference_case)
    else:
        from blackwing.reporting import render_markdown_report
        st.markdown(render_markdown_report(case["envelope"]))
    compiled = copilot.generate_report(case)
    if compiled["accepted_count"]:
        st.download_button("Download reviewed preliminary PDF", render_report_pdf(compiled),
                           file_name=f"BLACKWING-{case['incident']['case_id']}-preliminary.pdf", mime="application/pdf", key="export-pdf")
    else:
        st.info("Reviewed PDF export requires accepted findings with actual source excerpts. Backend JSON/Markdown exports remain available.")
    for format, label in (("json", "JSON"), ("md", "Markdown")):
        # Fetch exports only on explicit request, not whenever a hidden tab reruns.
        if st.button(f"Prepare {label} export", key=f"prepare-export-{format}"):
            data = copilot.export_report(case["incident"]["case_id"], format)
            st.download_button(f"Download {label}", data, file_name=f"BLACKWING-{case['incident']['case_id']}.{format}", key=f"download-{format}")
    with st.expander("Source agent work products"):
        st.json(report["agent_outputs"])


def show_saved_qa(case):
    st.subheader("Investigation conversation")
    st.caption("Browse answers already produced by the eight-agent backend. This view does not launch a separate model workflow.")
    query = st.text_input("Find information in this investigation", key="saved-question")
    if st.button("Find in saved results", key="find-saved"):
        roles = (case["report"] or {}).get("agent_outputs") or {}
        terms = set(query.casefold().split())
        selected = {key: value for key, value in roles.items() if terms and any(term in (key + json.dumps(value)).casefold() for term in terms)}
        st.session_state["saved_answer"] = {"case_id": case["incident"]["case_id"], "question": query, "outputs": selected}
    answer = st.session_state.get("saved_answer")
    if answer and answer["case_id"] == case["incident"]["case_id"]:
        with st.chat_message("user"):
            st.text(answer["question"])
        with st.chat_message("assistant"):
            if answer["outputs"]:
                st.json(answer["outputs"])
            else:
                st.info("No matching saved analysis. Missing evidence remains unresolved.")


def workspace(case, render_report):
    incident = case["incident"]
    st.header(case_name(incident))
    st.caption(f"Case ID · {incident['case_id']} · {incident['status']}")
    latest = max(case["runs"], key=lambda r: r["created_at"]) if case["runs"] else None
    running = latest and latest["status"] in {"queued", "running"}
    if latest:
        st.text(f"Investigation run · {latest['run_id']} · {latest['status']}")
        label = execution_label(latest.get("progress"))
        if label:
            st.text(label)
        if latest.get("error_code") == "interrupted":
            st.warning("The previous run was interrupted. Start a new recovery run; its history remains available.")
        elif latest["status"] == "failed":
            st.error(latest.get("error") or "Backend analysis failed.")
        if case["report_run_id"] and case["report_run_id"] != latest["run_id"]:
            st.warning(f"Displaying the earlier completed report {case['report_run_id']}; the latest run has no completed report.")
    include_retrieval = st.checkbox("Use existing RAG evidence", value=True, key="include-retrieval")
    if st.button("Run eight-agent investigation" if not latest else "Start a new investigation run", type="primary", disabled=bool(running), key="run-investigation"):
        copilot.start_investigation(incident["case_id"], include_retrieval=include_retrieval)
        st.rerun()
    if st.button("Refresh investigation status", key="refresh-case"):
        st.rerun()
    if running:
        st.caption("The backend continues independently. Refresh to retrieve progress; the UI never resubmits a run automatically.")
        st.json(latest.get("progress") or [])
    report = case["report"] or {}
    roles = report.get("agent_outputs") or {}
    tabs = st.tabs(["Investigation report", "Agent details", "Run trace"])
    with tabs[0]:
        quality_panel(incident, case["report"])
        show_report(case, render_report)
    with tabs[1]:
        st.subheader("Source agent work products")
        st.caption("Historical analogies cannot establish current-case findings.")
        for name, output in roles.items():
            with st.expander(name.replace("_", " ").title()):
                st.json(output)
        show_risk(report)
        # Preserve Sunayana's original evidence checklist and human review.
        show_evidence(case, include_quality=False)
        show_saved_qa(case)
    with tabs[2]:
        if case["runs"]:
            selected = st.selectbox("Inspect a backend run", [r["run_id"] for r in reversed(case["runs"])], key="history-run")
            if st.button("Load trace and audit", key="load-trace"):
                st.json(copilot.get_trace(selected))
                st.json(copilot.get_audit(selected))
        else:
            st.info("No backend runs yet.")
        st.caption("Backend SQLite is the sole durable history. Browser state contains only selections and an unsaved search answer.")


def main(*, brand, render_report, cloud=False):
    st.set_page_config(page_title="BLACKWING AI Copilot", page_icon="✈", layout="wide")
    brand()
    with st.sidebar:
        st.header("Case workspace")
        if st.button("Start New Investigation", width="stretch", key="nav-new"):
            st.session_state["workspace_view"] = "new"
        if st.button("Explore Historical Incidents", width="stretch", key="nav-history"):
            st.session_state["workspace_view"] = "history"
        st.caption("Inference and persistence are managed by FastAPI. No model API key is needed by this frontend.")
        status = copilot.backend_status()
        st.text(f"Backend status · {status['status']}")
        if cloud:
            st.caption("Cloud entrypoint uses the same backend; no temporary session database.")
    try:
        view = st.session_state.get("workspace_view", "home")
        if view == "home":
            st.title("BLACKWING AI Copilot")
            new, old = st.columns(2)
            with new:
                with st.container(border=True):
                    st.subheader("New investigation")
                    st.write("Enter an incident narrative and attach supporting records.")
                    if st.button("Start New Investigation", key="home-new", type="primary", width="stretch"):
                        st.session_state["workspace_view"] = "new"
                        st.rerun()
            with old:
                with st.container(border=True):
                    st.subheader("Historical incidents")
                    st.write("Reopen investigations saved in the backend repository.")
                    if st.button("Explore Historical Incidents", key="home-history", width="stretch"):
                        st.session_state["workspace_view"] = "history"
                        st.rerun()
        elif view == "new":
            register_case()
        elif view == "history":
            explore_cases()
        else:
            workspace(copilot.get_case(st.session_state["workspace_case"]), render_report)
    except httpx.HTTPStatusError as exc:
        # Do not echo provider/transport details or credentials into the UI.
        if exc.response.status_code == 409:
            st.error("The backend rejected this action because the case/run state changed. Refresh and try the permitted action.")
        else:
            st.error(f"The FastAPI backend returned HTTP {exc.response.status_code}. Check the case data and backend log.")
    except httpx.HTTPError:
        st.error("The FastAPI backend is unavailable. Start it and set BLACKWING_UI_API_URL to its address.")
    except (ValueError, UnicodeError) as exc:
        st.error(str(exc))


def show_gold_comparison(case):
    st.subheader('Gold comparison score')
    st.caption('Separate from structural completeness and aviation risk. Human review required.')
    try:
        result = copilot.get_gold_comparison(case['incident']['case_id'], case['report_run_id'])
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            st.info('Restart FastAPI with the updated code to enable gold comparison.')
            return
        raise
    reference_case = result.get('reference_case', bool(result.get('gold_id')))
    policy_mode = result.get('comparison_mode') == 'synthetic_safety_policy'
    if result.get('scope_notice'):
        st.caption(result['scope_notice'])
    if result.get('last_error'):
        st.error(result['last_error']['message'])
    if result['status'] in {'not_applicable','unavailable'}:
        st.metric('Gold comparison score', 'Not applicable' if result['status'] == 'not_applicable' else 'Unavailable')
        st.info(result['reason'])
        return reference_case
    st.caption('Matching reference: '+result.get('gold_case','')+' · '+result.get('gold_id',''))
    if result['status'] in {'not_requested','evaluation_incomplete'}:
        st.metric('Gold comparison score', 'Not assessed')
        if result['status'] == 'evaluation_incomplete':
            st.warning('Some comparisons need validation. No final numerical score has been assigned. '
                       'Validated judgments are preserved; retry checks only the remaining items.')
        button_label = ('Retry unvalidated comparisons' if result['status'] == 'evaluation_incomplete'
                        else 'Compare with gold evaluation' if policy_mode else 'Compare with gold report')
        if st.button(button_label, key='compare-gold-'+case['report_run_id']):
            try:
                with st.spinner('Comparing the completed report with its matching gold reference…'):
                    result = copilot.compare_with_gold(case['incident']['case_id'],case['report_run_id'])
            except httpx.HTTPStatusError as exc:
                try:
                    detail = exc.response.json().get('detail')
                except ValueError:
                    detail = None
                message = detail.get('message') if isinstance(detail,dict) else detail
                st.error(message or f'Gold comparison failed (HTTP {exc.response.status_code}). Inspect the backend log; the report is preserved.')
                return reference_case
            except httpx.HTTPError:
                st.error('Gold comparison connection failed. The report is preserved; refresh before retrying.')
                return reference_case
            st.rerun()
    else:
        st.metric('Gold comparison score', f"{result['score']:g}/100" if result.get('score') is not None else 'Not assessable')
    if result.get('reason'):
        st.info(result['reason'])
    if result.get('comparisons'):
        rows = result['comparisons']
        validated = sum(not bool(row.get('validation_error')) for row in rows)
        st.caption(f"Validated comparisons: {validated}/{len(rows)}. Pending: {len(rows)-validated}. "
                   'An evaluator failure does not mean the report claim is false.')
    st.caption(result.get('method',''))
    st.info(result.get('notice','Human review required.'))
    if policy_mode:
        st.caption(f"Applicable safety policies compared: {result.get('criteria_compared',0)}; not applicable: {result.get('criteria_not_applicable',0)}. Incident facts were not gold-validated.")
    else:
        st.caption(f"Factual claims compared: {result.get('factual_claims',0)}. Run: {case['report_run_id']}")
    if result.get('comparisons'):
        st.dataframe(result['comparisons'], hide_index=True, alt='Claim comparisons with matching gold-page quotations')
    if result.get('improvements'):
        with st.expander('Improvements to review'):
            for item in result['improvements']:
                st.write(item['report_field']+' — '+item['issue'])
                st.write(item['action'])
    st.download_button('Download gold comparison', json.dumps(result,indent=2).encode(),
        file_name='BLACKWING-gold-comparison-'+case['report_run_id']+'.json',mime='application/json',
        key='download-gold-'+case['report_run_id'])

    return reference_case
