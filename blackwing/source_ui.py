"""Sunayana's original investigation form; shared backend, investigation and Q&A tabs."""
from blackwing import copilot
from blackwing.copilot_state import execution_label, report_availability, risk_display
from blackwing.copilot_ui import quality_panel, report_completeness_panel, show_evidence, show_gold_comparison, show_final_report_evaluation_trace, show_final_report_gold_action
import httpx
import streamlit as st


def render_case(case, render_report):
    incident=case['incident']
    latest=max(case['runs'],key=lambda r:r['created_at']) if case['runs'] else None
    report=case.get('report')
    partial=case.get('partial_report')
    st.markdown('## Investigation run')
    if latest:
        st.caption(f"Case ID: {incident['case_id']} · Run ID: {latest['run_id']} · Status: {latest['status']}")
        label=execution_label(latest.get('progress'))
        if label:
            st.text(label)
        if latest.get('error_code')=='interrupted':
            st.warning('The previous investigation was interrupted. Its trace is preserved.')
        elif latest['status']=='failed':
            st.error(latest.get('error') or 'Investigation failed; inspect Run trace for the failed agent.')
        if case.get('report_run_id') and case['report_run_id']!=latest['run_id']:
            st.warning('Displaying the earlier completed report; the latest run has no completed report.')
    report_tab,agents_tab,trace_tab=st.tabs(['Investigation report','Agent details','Run trace'])
    with report_tab:
        if latest and latest['status'] in {'queued','running'}:
            loader=st.status('Generating the complete investigation report…',state='running',expanded=True)
            loader.write(execution_label(latest.get('progress')) or
                         'The report will appear here when all required sections are ready.')
            if report and case.get('report_run_id') != latest['run_id']:
                st.info('A previous completed report is available below while this run finishes.')
                with st.expander('Previous completed report',expanded=False):
                    reference=(report.get('agent_outputs',{}).get('similar_case',{}).get('case_status')=='existing_case_match')
                    if report.get('generated_report'):
                        render_report(report['generated_report'],report,incident['case_id'],case['report_run_id'],reference_case=reference)
                    else:
                        from blackwing.reporting import render_markdown_report
                        st.markdown(render_markdown_report(case['envelope']))
            else:
                st.info('No partial report is shown. The completed report will appear automatically.')
        elif not report:
            availability=case.get('report_availability') or report_availability(case['runs'],None)
            pending='Pending' if availability['state'] in {'running','not_started'} else 'Unavailable'
            risk=(partial or {}).get('agent_outputs',{}).get('risk')
            if risk:
                rows=risk_display(partial)['rows']
                indices=list(dict.fromkeys(row['Risk index'] for row in rows if row['Risk index']!='Unassessed'))
                st.metric('Aviation risk',' · '.join(indices) if indices else 'Unassessed')
            else:
                st.metric('Aviation risk',pending)
            if availability['state'] in {'failed','interrupted','unavailable'}:
                st.warning(availability['message'])
            elif availability['state']!='running':
                st.info(availability['message'])
        else:
            report_completeness_panel(incident,report)
            gold_match=show_gold_comparison(case)
            if not gold_match:
                show_final_report_gold_action(case)
            show_final_report_evaluation_trace(case)
            reference=(report.get('agent_outputs',{}).get('similar_case',{}).get('case_status')=='existing_case_match')
            if report.get('generated_report'):
                render_report(report['generated_report'],report,incident['case_id'],case['report_run_id'],reference_case=reference)
            else:
                from blackwing.reporting import render_markdown_report
                st.markdown(render_markdown_report(case['envelope']))
    with agents_tab:
        st.markdown('### Source agent work products')
        st.caption('Structured JSON stays available for review and citation checking.')
        outputs=(report or partial or {}).get('agent_outputs',{})
        for name,output in outputs.items():
            with st.expander(name.replace('_',' ').title()):
                st.json(output)
        if report:
            with st.expander('Evidence checklist and human review'):
                show_evidence(case,include_quality=False)
    with trace_tab:
        st.markdown('### Run trace')
        if latest:
            from blackwing.evaluation_ui import show_run_trace
            show_run_trace(latest, report or partial)
        else:
            st.info('Run an investigation to see its workflow timeline and telemetry.')


def _render_live_case(case_id, render_report):
    try:
        case = copilot.get_case(case_id)
        latest = max(case['runs'], key=lambda row: row['created_at']) if case['runs'] else None
        if latest and case.get('report_run_id') != latest['run_id']:
            audit = copilot.get_audit(latest['run_id'])
            partial_outputs = dict(audit.get('partial_outputs') or {})
            generated = partial_outputs.pop('report',None)
            if partial_outputs:
                case['partial_report'] = {'agent_outputs':partial_outputs,'generated_report':generated}
        was_running = st.session_state.get('source_run_active', False)
        running = bool(latest and latest['status'] in {'queued', 'running'})
        st.session_state['source_run_active'] = running
        render_case(case, render_report)
        if was_running != running:
            st.rerun()  # Align polling interval and submit state with backend status.
    except httpx.HTTPError:
        st.error('Cannot read investigation status from ' + copilot.backend_url() +
                 '. The backend may be stopped or busy. Your submitted case is preserved; reconnect before retrying Analyze.')


def main(*,brand,render_report,cloud=False):
    st.set_page_config(page_title='BlackWing AI',page_icon='✈️',layout='wide')
    brand()
    default_tab = 'Q&A' if st.query_params.get('view') == 'qa' else 'Start an investigation'
    investigation_tab, qa_tab = st.tabs(
        ['Start an investigation', 'Q&A'], default=default_tab,
        key='workspace-tabs', on_change='rerun')
    with investigation_tab:
        if investigation_tab.open:
            _investigation(render_report, cloud=cloud)
    with qa_tab:
        if qa_tab.open:
            from blackwing.qa_ui import main as qa_page
            qa_page(embedded=True)


def _investigation(render_report, *, cloud=False):
    if cloud:
        st.caption('Cloud entrypoint uses the same backend; no temporary session database.')
    with st.form('incident_form'):
        st.markdown('<div class="bw-kicker">Start an investigation</div>',unsafe_allow_html=True)
        narrative=st.text_area('Incident narrative',value='',
            key='incident_narrative_custom',height=210,
            placeholder='Describe the event and include any known date, airport, aircraft, and source references.')
        uploaded=st.file_uploader('Attach source report (optional)',type=['pdf','txt'],max_upload_size=5,key='source-upload',
            help='Uploaded text is attached to this case as unverified evidence; it is not added to the shared RAG index.')
        submitted=st.form_submit_button('Analyze incident',type='primary',key='analyze-incident',
            disabled=st.session_state.get('source_run_active',False))
    try:
        if submitted:
            if len(narrative.strip())<20:
                st.error('Enter at least 20 characters of incident narrative.')
                return
            evidence=copilot.file_evidence([(uploaded.name,uploaded.getvalue())] if uploaded else [],
                'User supplied source: '+uploaded.name if uploaded else '')
            with st.spinner("Creating the case and submitting the investigation…"):
                incident=copilot.create_case(narrative=narrative,evidence=evidence)
            st.session_state['source_case_id']=incident['case_id']
            st.query_params['case_id']=incident['case_id']
            # Exactly one explicit Analyze click starts one backend run.
            st.session_state['source_run_active']=True
            with st.spinner('Submitting the background run…'):
                copilot.start_investigation(incident['case_id'])
            st.rerun()
        case_id=st.query_params.get('case_id') or st.session_state.get('source_case_id')
        # Preserve canonical saved-case bookmarks and existing workspace selections.
        if not case_id:
            case_id=st.session_state.get('workspace_case') if st.session_state.get('workspace_view')=='case' else None
        if case_id:
            refresh = '3s' if st.session_state.get('source_run_active',True) else None
            st.fragment(_render_live_case,run_every=refresh)(case_id,render_report)
    except httpx.HTTPStatusError as exc:
        st.error(f'FastAPI returned HTTP {exc.response.status_code}. Check the persisted case/run state before retrying.')
    except httpx.HTTPError:
        st.error('Cannot reach FastAPI at '+copilot.backend_url()+'. Start the backend in its configured IIT terminal. Do not resubmit until its health endpoint responds.')
    except (ValueError,UnicodeError) as exc:
        st.error(str(exc))
