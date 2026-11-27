"""Question evaluation inside the existing report panel; no generation or scoring."""
import json
from datetime import datetime

import httpx
import streamlit as st

from blackwing import copilot


def _event_time(event):
    value = event.get('timestamp') or ''
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return parsed.timestamp()
    except ValueError:
        return 0.0


def _trace_events(trace, audit):
    """Merge public trace and audit records without rendering duplicate spans."""
    events, seen = [], set()
    for event in [*(trace.get('events') or []), *(audit.get('events') or [])]:
        identity = event.get('event_id') or event.get('invocation_id')
        if identity is None:
            identity = (event.get('timestamp'), event.get('step'), event.get('status'),
                        event.get('agent_key') or event.get('agent'), event.get('tool'))
        if identity in seen:
            continue
        seen.add(identity)
        events.append(event)
    return sorted(events, key=_event_time)


def show_run_trace(run, report=None):
    """Render operational telemetry as a readable dashboard, not raw JSON."""
    run_id = str(run.get('run_id') or '')
    if not run_id:
        st.info('No run is selected.')
        return
    case_id = str(run.get('case_id') or '')
    selected_run_id = str(run_id)
    retrieval_key = 'retrieval-evaluation-v6-' + case_id + '-' + selected_run_id

    state_key = 'run-trace-snapshot-' + run_id
    if st.button('Refresh run trace', key='refresh-run-trace-' + run_id):
        st.session_state.pop(state_key, None)
        st.session_state.pop(retrieval_key, None)
    snapshot = st.session_state.get(state_key)
    if snapshot is None:
        with st.spinner('Loading run timeline and audit details…'):
            try:
                snapshot = {'trace': copilot.get_trace(run_id), 'audit': copilot.get_audit(run_id)}
                st.session_state[state_key] = snapshot
            except httpx.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                st.error(f'Could not load run telemetry{f" (HTTP {status})" if status else ""}. The run and report are preserved.')
                return

    trace = snapshot.get('trace') or {}
    audit = snapshot.get('audit') or {}
    events = _trace_events(trace, audit)
    metrics = audit.get('operations_metrics') or {}
    tool_events = [event for event in events if event.get('tool')]
    agent_events = [event for event in events if event.get('agent_key') or event.get('agent')]
    errors = [event for event in events if event.get('error') or event.get('status') in {'failed', 'agent_failed', 'model_request_failed', 'execution_failed'}]

    st.subheader('Run overview')
    columns = st.columns(5)
    columns[0].metric('Run status', str(audit.get('status') or run.get('status') or 'unknown').replace('_', ' ').title())
    latency = metrics.get('latency_ms')
    columns[1].metric('End-to-end latency', f'{latency / 1000:.1f} s' if isinstance(latency, (int, float)) else 'In progress' if run.get('status') in {'queued', 'running'} else 'Unavailable')
    columns[2].metric('Agents involved', len({event.get('agent_key') or event.get('agent') for event in agent_events}))
    columns[3].metric('Tool calls', metrics.get('tool_calls', len(tool_events)))
    tokens = (metrics.get('prompt_tokens') or 0) + (metrics.get('completion_tokens') or 0)
    columns[4].metric('Model tokens', f'{tokens:,}' if tokens else 'Unavailable')
    cost = metrics.get('estimated_cost_usd')
    st.caption(f"Model: {audit.get('model') or run.get('model') or 'not recorded'} · "
               f"Execution steps: {audit.get('execution_steps', '—')} · "
               f"Estimated cost: ${cost:.4f}" if isinstance(cost, (int, float)) else
               f"Model: {audit.get('model') or run.get('model') or 'not recorded'} · Execution steps: {audit.get('execution_steps', '—')} · Cost rates are not configured")

    if not events:
        st.info('No trace events were recorded for this run.')
        return

    st.subheader('Execution timeline')
    display_rows = []
    for index, event in enumerate(events, 1):
        error = event.get('error') or {}
        timestamp = event.get('timestamp') or ''
        display_rows.append({
            '#': event.get('sequence') or index,
            'Time': str(timestamp).replace('T', ' ')[:23],
            'Stage': str(event.get('step') or event.get('event') or 'workflow').replace('_', ' ').title(),
            'Agent / tool': event.get('agent') or event.get('agent_key') or event.get('tool') or '—',
            'Status': str(event.get('status') or 'recorded').replace('_', ' ').title(),
            'Duration (ms)': event.get('duration_ms'),
            'Attempt': event.get('attempt'),
            'Evidence count': event.get('retrieved_document_count') or
                              (event.get('result_summary') or {}).get('count'),
            'Error': error.get('code') or error.get('type') or '',
        })
    st.dataframe(display_rows, hide_index=True, width='stretch',
                 alt='Chronological agent, retrieval, tool, and workflow events for this run')

    if errors:
        grouped = {}
        for event in errors:
            error = event.get('error') or {}
            label = error.get('code') or error.get('type') or event.get('status') or 'unknown'
            grouped[label] = grouped.get(label, 0) + 1
        st.subheader('Failures and retries')
        st.dataframe([{'Failure': name.replace('_', ' '), 'Occurrences': count}
                      for name, count in sorted(grouped.items())], hide_index=True,
                     width='stretch', alt='Recorded failures grouped by error code')

    st.subheader('Event details')
    for index, event in enumerate(events, 1):
        title = (f"{event.get('sequence') or index}. "
                 f"{str(event.get('step') or event.get('event') or 'event').replace('_', ' ')} · "
                 f"{str(event.get('status') or 'recorded').replace('_', ' ')}")
        with st.expander(title, expanded=False):
            detail = event.get('detail')
            if detail:
                if not isinstance(detail, str):
                    detail = json.dumps(detail, ensure_ascii=False, default=str)
                st.write(detail[:5000] + ('…' if len(detail) > 5000 else ''))
            summary = event.get('result_summary')
            if summary:
                st.write('Result summary')
                st.json(summary)
            identifiers = event.get('input_identifiers')
            if identifiers:
                st.caption('Evidence identifiers: ' + ', '.join(map(str, identifiers)))
            error = event.get('error')
            if error:
                safe_error = {key: error.get(key) for key in
                              ('code', 'type', 'message', 'http_status', 'json_error', 'validation_fields')
                              if error.get(key) is not None}
                st.error(safe_error or 'The event recorded an error.')
            for field, label in (('request_id', 'Request ID'), ('execution_id', 'Execution ID'),
                                 ('invocation_id', 'Tool invocation ID')):
                if event.get(field):
                    st.caption(f'{label}: {event[field]}')

    with st.expander('Workflow and review notes', expanded=False):
        plan = audit.get('plan')
        if plan:
            st.write('Saved workflow plan')
            if isinstance(plan, (dict, list)):
                st.json(plan)
            else:
                st.write(str(plan))
        limitations = (report or {}).get('limitations') or []
        for limitation in limitations:
            st.write('• ' + str(limitation))
        st.info('AI output is preliminary until a qualified human reviewer records a decision.')

    case_id = (run.get('case_id') or '')
    if case_id:
        st.subheader('Evaluation activity')
        st.caption('This trace shows evaluation status and coverage. Report completeness and gold-comparison scores stay in the Investigation report.')
        evaluations = []
        run_status = str(audit.get('status') or run.get('status') or '').lower()
        if run_status != 'completed':
            evaluations.extend([
                {'Evaluation': 'Final report section evaluation', 'Status': 'waiting',
                 'Coverage': 'Available after the investigation report completes'},
                {'Evaluation': 'Question-based report evaluation', 'Status': 'waiting',
                 'Coverage': 'Available after the investigation report completes'},
            ])
        else:
            try:
                final_eval = copilot.get_final_report_evaluation(case_id, run_id)
                sections = final_eval.get('sections') or {}
                scored = sum(bool((row.get('llm_judge') or {}).get('score') is not None) for row in sections.values())
                semantic = sum(bool(row.get('cosine') or row.get('bertscore')) for row in sections.values())
                evaluations.append({'Evaluation': 'Final report section evaluation',
                                    'Status': str(final_eval.get('status') or 'unknown').replace('_', ' '),
                                    'Coverage': f'{scored}/{final_eval.get("sections_total", len(sections))} judge sections · {semantic}/{len(sections)} with semantic metrics'})
            except httpx.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                waiting = status == 409
                evaluations.append({'Evaluation': 'Final report section evaluation',
                                    'Status': 'waiting' if waiting else 'unavailable',
                                    'Coverage': 'Available after the investigation report completes' if waiting else 'Could not load evaluation status'})
            try:
                report_eval = copilot.get_report_evaluation(case_id, run_id)
                total = report_eval.get('questions_total', 0)
                completed = report_eval.get('questions_completed', 0)
                evaluations.append({'Evaluation': 'Question-based report evaluation',
                                    'Status': str(report_eval.get('status') or 'unknown').replace('_', ' '),
                                    'Coverage': f'{completed}/{total} judgments validated' if total else 'No question set configured'})
            except httpx.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                waiting = status == 409
                evaluations.append({'Evaluation': 'Question-based report evaluation',
                                    'Status': 'waiting' if waiting else 'unavailable',
                                    'Coverage': 'Available after the investigation report completes' if waiting else 'Could not load evaluation status'})
        st.dataframe(evaluations, hide_index=True, width='stretch',
                     alt='Evaluation stage status and completed coverage; numeric scores are displayed in the investigation report')

    st.subheader('Retrieval benchmark')
    if not case_id:
        st.info('Retrieval recall is unavailable because this run is not linked to a case.')
        return
    st.caption('Recall@5/@10 = relevant sources found in the top 5/10 retrieved sources ÷ all relevant sources. Relevance comes from approved labels when the case has them, otherwise from a rule (applicable to the case date and jurisdiction, and about the incident category). The incident ID is never used as a query.')
    # Bind cached metrics to this exact run; a new run creates a fresh evidence set.
    retrieval_eval = st.session_state.get(retrieval_key)
    if retrieval_eval is None:
        try:
            retrieval_eval = copilot.get_retrieval_evaluation(case_id, selected_run_id)
            # Cache completed measurements only. A transient not-measurable
            # response must not stick in the UI after run state is updated.
            if retrieval_eval.get('status') == 'completed':
                st.session_state[retrieval_key] = retrieval_eval
        except httpx.HTTPError:
            retrieval_eval = {'status': 'unavailable', 'reason': 'Could not load retrieval results for this case.'}
    if retrieval_eval.get('status') == 'completed':
        cols = st.columns(2)
        cols[0].metric('Recall@5', f"{retrieval_eval['recall@5']:.1%}")
        cols[1].metric('Recall@10', f"{retrieval_eval['recall@10']:.1%}")
        st.caption(f"Run: {retrieval_eval.get('run_id') or selected_run_id} · "
                   f"{retrieval_eval.get('query_count', 0)} evaluation query/queries · "
                   f"retrieval method: {retrieval_eval.get('method', 'hybrid')} · "
                   f"{retrieval_eval.get('relevant_source_count', 0)} relevant sources")
        st.dataframe([{
            'Evaluation basis': row.get('query_id') or 'Approved question',
            'Relevant sources': row.get('relevant_source_count'),
            'Found @5': f"{row.get('relevant_found@5', 0)} / {row.get('relevant_source_count', 0)}",
            'Recall@5': f"{row.get('recall@5', 0):.1%}",
            'Found @10': f"{row.get('relevant_found@10', 0)} / {row.get('relevant_source_count', 0)}",
            'Recall@10': f"{row.get('recall@10', 0):.1%}",
        } for row in retrieval_eval.get('queries', [])], hide_index=True, width='stretch',
            alt='Recall against approved relevant sources for each labeled question')
        later_sources = retrieval_eval.get('temporally_ineligible_sources') or []
        if later_sources:
            labels = ', '.join(
                f"{item.get('source_id')} ({item.get('source_date')}: {item.get('reason')})"
                for item in later_sources
            )
            st.warning(
                'This run\'s evidence included sources dated on or after the case, which are not valid '
                f"historical predecessors: {labels}. They are excluded from the recall calculation above."
            )
        st.caption(retrieval_eval.get('notice', ''))
    else:
        reason = retrieval_eval.get('reason') or 'Approved retrieval labels are unavailable for this case.'
        if retrieval_eval.get('status') == 'not_measurable':
            cols = st.columns(2)
            cols[0].metric('Recall@5', 'Not measurable')
            cols[1].metric('Recall@10', 'Not measurable')
            st.info(reason)
        else:
            st.info(reason)


def show_evaluation_options():
    """Explain complementary checks without presenting uncomputed scores."""
    with st.expander('Other evaluation options'):
        st.markdown(
            '- **Claim grounding:** check each report claim against its cited passage; '
            'label supported, partial, unsupported or not verifiable.\n'
            '- **Section rubric:** score chronology, attribution, uncertainty, hazards, '
            'mitigations and required unknowns against a short rubric.\n'
            '- **Retrieval quality:** measure recall@k and precision against human-labeled '
            'relevant passages, separately from report quality.\n'
            '- **Safety checks:** deterministic tests for invented citations, protected-gold '
            'leakage, unsupported causal claims and jurisdiction mismatches.\n'
            '- **Workflow metrics:** task completion, retries, tool calls, latency and token use.'
        )
        st.caption('Only evaluators shown in this Run trace have scores. The other checks are recommendations, not hidden scores.')


def show_question_evaluation_trace(result):
    """Read-only evaluation detail for Run trace; actions stay in the report view."""
    st.subheader('Gold comparison')
    if not result:
        st.info('Gold comparison status is unavailable for this run.')
        return
    if result.get('reference_id'):
        st.caption('Approved reference · ' + str(result['reference_id']))
    status = result.get('status', 'unavailable')
    score = result.get('score')
    final_score = status in {'completed', 'proposed'} and type(score) in (int, float) and 0 <= score <= 100
    st.metric('Gold comparison score', f'{score:g}/100' if final_score else
              'Unavailable' if status == 'unavailable' else
              'Not assessable' if status == 'not_assessable' else 'Not assessed')
    total = result.get('questions_total', 0)
    completed = result.get('questions_completed', 0)
    if total:
        st.caption(f'Status: {status.replace("_", " ")} · Validated: {completed}/{total} · '
                   f'Scored: {result.get("questions_scored", 0)} · '
                   f'Excluded: {result.get("questions_excluded", 0)} · '
                   f'Needs review: {result.get("questions_requires_review", 0)} · '
                   f'Failed: {result.get("questions_failed", 0)}')
    else:
        st.caption('Status: ' + status.replace('_', ' '))
    if result.get('reason'):
        st.info(result['reason'])
    questions = {q['question_id']: q['question'] for q in result.get('questions', [])}
    rows = result.get('judgments') or []
    if rows:
        st.dataframe([
            {'question_id': row.get('question_id'),
             'question': questions.get(row.get('question_id'), ''),
             **{key: row.get(key) for key in ('status', 'points', 'report_field',
                                               'report_quote', 'reference_page',
                                               'reference_quote', 'rationale', 'error_code')}}
            for row in rows
        ], hide_index=True, alt='Gold comparison judgments and report/reference evidence')
    comparisons = result.get('comparisons') or []
    if comparisons:
        st.dataframe(comparisons, hide_index=True,
                     alt='Gold comparison claim judgments and supporting quotations')
    if result.get('improvements'):
        with st.expander('Evaluation findings to review'):
            for item in result['improvements']:
                st.write(str(item.get('report_field') or 'Report') + ' — ' +
                         str(item.get('issue') or 'Review this finding.'))
                st.write(item.get('action') or 'Human review required.')


def show_final_report_evaluation_trace(case, result):
    """Show section-level final-report metrics and explicit evaluation control."""
    st.subheader('Final report evaluation')
    st.markdown('**LLM-as-a-Judge · Semantic similarity (BGE cosine similarity/distance) · BERTScore**')
    if not result:
        st.info('Final report evaluation status is unavailable.')
        return
    status = result.get('status', 'unavailable')
    st.caption(result.get('notice') or 'Gold reference is evaluator-only and excluded from report generation and RAG.')
    if result.get('reason'):
        st.info(result['reason'])
    if result.get('reference_id'):
        st.caption(f"Reference: {result['reference_id']} · {result.get('reference_page_count', '—')} pages · section-level comparison")
    if result.get('sections_not_available'):
        st.caption('No generated section available: ' + ', '.join(result['sections_not_available']))

    if status in {'not_requested', 'interrupted', 'failed', 'incomplete'}:
        from blackwing import copilot
        label = 'Run final report evaluations' if status == 'not_requested' else 'Retry final report evaluations'
        if st.button(label, key='run-final-report-evals-' + str(result.get('run_id') or case['report_run_id'])):
            try:
                copilot.start_final_report_evaluation(case['incident']['case_id'], case['report_run_id'])
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 409:
                    st.error('Wait for the active investigation to finish, or check that a unique matching gold report exists in the configured directory.')
                elif exc.response.status_code == 503:
                    st.error('The evaluator-only gold directory is unavailable. Check the API environment and directory contents.')
                else:
                    st.error(f'Final report evaluation could not start (HTTP {exc.response.status_code}).')
            except httpx.HTTPError:
                st.error('Could not reach the API to start final report evaluation.')
            else:
                st.rerun()
    elif status in {'queued', 'running'}:
        st.info(f"Evaluation {status}. Section results are saved as each section completes.")
    elif status == 'not_applicable':
        st.caption('No matching gold reference applies to this saved run, so no score is produced.')
    elif status == 'unavailable':
        st.caption('Configure the evaluator-only gold directory in the API environment to enable section scoring.')
        st.button('Run final report evaluations', disabled=True,
                  help='Set BLACKWING_FINAL_REPORT_GOLD_DIR and restart FastAPI.',
                  key='final-report-evals-disabled-' + str(result.get('run_id') or case.get('report_run_id') or 'none'))
    elif status == 'not_assessable':
        st.caption('No unique matching gold PDF was found, so unrelated reports were not used.')
        st.button('Run final report evaluations', disabled=True,
                  help='Add a uniquely matching PDF to the configured gold directory.',
                  key='final-report-evals-unmatched-' + str(result.get('run_id') or case.get('report_run_id') or 'none'))

    overall = result.get('overall') or {}
    if result.get('metric_warnings'):
        st.caption('Some metrics could not be computed: ' + ', '.join(
            f"{item.get('metric')} ({item.get('error_type')})" for item in result['metric_warnings']))
    if overall:
        metrics = st.columns(3)
        with metrics[0]:
            value = overall.get('llm_judge_score')
            st.metric('LLM judge', f'{value:g}/100' if value is not None else 'Unavailable')
        with metrics[1]:
            value = overall.get('cosine_similarity_percent')
            st.metric('Cosine similarity', f'{value:g}%' if value is not None else 'Unavailable')
            distance = overall.get('cosine_distance')
            if distance is not None:
                st.caption(f'Cosine distance: {distance:.4f}')
        with metrics[2]:
            value = overall.get('bertscore_f1_percent')
            st.metric('BERTScore F1', f'{value:g}%' if value is not None else 'Unavailable')
        st.caption(overall.get('aggregation', 'Metrics are reported separately; they are not combined.'))

    sections = result.get('sections') or {}
    if sections:
        rows = []
        for key, item in sections.items():
            judge = item.get('llm_judge') or {}
            cosine = item.get('cosine') or {}
            bert = item.get('bertscore') or {}
            rows.append({
                'Report section': item.get('title') or key,
                'LLM judge / 100': judge['score'] if judge.get('score') is not None else 'Unavailable',
                'Coverage / 100': judge['coverage'] if judge.get('coverage') is not None else 'Unavailable',
                'Consistency / 100': judge['consistency'] if judge.get('consistency') is not None else 'Unavailable',
                'Cosine similarity': cosine.get('cosine_similarity'),
                'Cosine distance': cosine.get('cosine_distance'),
                'BERTScore F1': bert.get('f1'),
                'Status': item.get('status'),
            })
        st.dataframe(rows, hide_index=True, use_container_width=True,
                     alt='Final report evaluation scores by published report section')
        for key, item in sections.items():
            with st.expander(item.get('title') or key):
                st.caption('Gold section used: ' + str(item.get('reference_scope') or ''))
                judge = item.get('llm_judge') or {}
                if judge.get('rationale'):
                    st.write(judge['rationale'])
                if judge.get('material_omissions'):
                    st.markdown('**Material omissions identified by judge**')
                    for line in judge['material_omissions']:
                        st.write('• ' + line)
                if judge.get('unsupported_or_conflicting_claims'):
                    st.markdown('**Unsupported or conflicting claims**')
                    for line in judge['unsupported_or_conflicting_claims']:
                        st.write('• ' + line)
                if item.get('judge_error'):
                    st.caption('LLM judge unavailable for this section: ' + str(item['judge_error']))
    if status in {'queued', 'running'}:
        st.caption('Refresh this tab to load newly completed section scores.')


def show_question_evaluation(case, result):
    st.subheader('Gold comparison')
    st.caption('LLM judge against an approved reference, when available. Human review required.')
    if result.get('reference_id'):
        st.caption('Matching reference: ' + result['reference_id'])
    if result.get('evaluation_split') == 'held_out_reference_case':
        st.caption('Held-out reference comparison. Having a matching reference does not establish prior development use of this case.')
    status = result.get('status', 'unavailable')
    score = result.get('score')
    final_score = status == 'completed' and type(score) in (int, float) and 0 <= score <= 100
    st.metric('Gold comparison score', f'{score:g}/100' if final_score else
              'Unavailable' if status == 'unavailable' else
              'Not assessable' if status == 'not_assessable' else 'Not assessed')
    st.caption('Comparison status: ' + status.replace('_', ' '))
    if status == 'unavailable':
        st.info(result.get('reason') or 'Matching approved questions and reference are unavailable.')
        return False
    if status == 'not_assessable':
        st.info(result.get('reason') or 'Question-based gold comparison is unavailable for this report.')
        return False

    total = result.get('questions_total', 0)
    completed = result.get('questions_completed', 0)
    failed = result.get('questions_failed', 0)
    if total:
        st.caption(f'Validated questions: {completed}/{total}. Scored: {result.get("questions_scored", 0)}. '
                   f'Excluded: {result.get("questions_excluded", 0)}. '
                   f'Needs review: {result.get("questions_requires_review", 0)}. Failed: {failed}.')
        st.progress(min(max(completed / total, 0), 1))

    if status in {'queued', 'running'}:
        events = result.get('events') or []
        waiting = events and events[-1].get('event') == 'waiting_for_model_response'
        current = events[-1].get('question_id') if events else None
        progress = f' Working on {current}.' if current else ''
        st.info(('Waiting for model response.' if waiting else 'Question evaluation is queued or running.') +
                progress + ' Status updates automatically every few seconds.')
    elif status == 'interrupted':
        st.warning('Evaluation was interrupted. Saved validated judgments are preserved; retry resumes eligible questions.')
    elif status in {'failed', 'incomplete'}:
        st.warning('Evaluation did not finish. No final numerical score has been assigned. '
                   'The investigation report and validated judgments are preserved.')
    elif status == 'requires_review':
        st.warning('Some judgments require review or contain unsupported conclusions. '
                   'No final numerical score has been assigned.')
    elif status == 'not_assessable':
        st.info('No scoreable question coverage remains. Justified abstention and inapplicable questions are excluded; no score is invented.')

    case_id, run_id = case['incident']['case_id'], case['report_run_id']
    if status in {'not_requested', 'interrupted', 'failed', 'incomplete'}:
        label = 'Compare with gold report' if status == 'not_requested' else 'Retry unvalidated comparisons'
        if st.button(label, key='compare-gold-' + run_id):
            try:
                copilot.start_report_evaluation(case_id, run_id)
            except httpx.HTTPStatusError as exc:
                messages = {
                    409: 'Wait for the active investigation to finish before starting comparison.',
                    404: 'This saved run or evaluation endpoint is unavailable. Refresh the case and restart FastAPI with the updated code.',
                    503: 'Matching approved evaluation questions/reference could not be verified. Check the backend configuration.',
                }
                st.error(messages.get(exc.response.status_code,
                         'The backend could not start comparison. Check its log; the investigation report is preserved.'))
            except httpx.HTTPError:
                st.error('Comparison submission could not be confirmed. The report is preserved; refresh status before retrying.')
            else:
                st.rerun()
    if st.button('Refresh comparison status', key='refresh-gold-' + run_id):
        st.rerun()

    st.caption(result.get('method', 'Points are calculated by the evaluator backend after question judgments validate.'))
    st.info(result.get('notice', 'Human review required. An evaluator failure does not prove that a report statement is false.'))
    rows = result.get('judgments') or []
    if rows:
        questions = {q['question_id']: q['question'] for q in result.get('questions', [])}
        display = [{'question_id': r['question_id'], 'question': questions.get(r['question_id'], ''),
                    **{k: r.get(k) for k in ('status', 'points', 'report_field', 'report_quote',
                                           'reference_page', 'reference_quote', 'input_answerability',
                                           'input_id', 'input_quote', 'rationale', 'error_code', 'duration_seconds')}}
                   for r in rows]
        st.dataframe(display, hide_index=True, alt='Question judgments, supporting report/reference quotations and review needs')
    if result.get('improvements'):
        with st.expander('Improvements to review'):
            for item in result['improvements']:
                st.text(str(item.get('question_id') or '') + ' · ' + str(item.get('report_field') or 'Report review'))
                st.text(str(item.get('issue') or 'Review the evidence for this judgment.'))
                st.text(str(item.get('action') or 'Human review required.'))
    if status != 'not_requested':
        st.download_button('Download gold comparison', json.dumps(result, indent=2).encode(),
                           file_name='BLACKWING-gold-comparison-' + run_id + '.json',
                           mime='application/json', key='download-gold-' + run_id)
    return result.get('reference_case', True)
