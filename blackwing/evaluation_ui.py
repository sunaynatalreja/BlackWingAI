"""Question evaluation inside the existing report panel; no generation or scoring."""
import json

import httpx
import streamlit as st

from blackwing import copilot


def show_question_evaluation(case, result):
    st.subheader('Gold comparison score')
    st.caption('LaaJ (LLM-as-a-Judge) / gold evaluation. Separate from structural completeness and aviation risk. Human review required.')
    st.caption('Question-based comparison of the saved report against its matching approved reference.')
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
        return result.get('reference_case')

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
