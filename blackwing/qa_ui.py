"""Separate Q&A evaluation page; no investigation layout or local database."""
import json
import httpx
import streamlit as st
from blackwing import copilot


def main(*, embedded=False):
    if not embedded:
        st.title('Q&A')
        st.link_button('Back to investigation', '?view=investigation')
    st.caption('Generate an answer from the existing RAG index, then compare with the preserved gold answer. Human verification required.')
    try:
        with st.form('qa-question-form'):
            question = st.text_area('Enter your question', key='qa-question', max_chars=4000, height=150)
            ask = st.form_submit_button('Get answer', key='qa-generate', type='primary')
        if ask:
            if len(question.strip()) < 3:
                st.error('Enter a question before requesting an answer.')
                return
            with st.spinner('Retrieving evidence and generating one answer…'):
                result = copilot.request('POST', '/api/v1/qa/answers', json={'question':question})
            st.session_state['qa-answer-id'] = result['answer_id']
            st.rerun()
        history = copilot.request('GET', '/api/v1/qa/answers')
        reviewed = sum(bool(r.get('review')) for r in history)
        st.caption(f'Saved answers: {len(history)}. Human reviewed: {reviewed}. Questions run only when you submit them.')
        if history:
            with st.expander('Saved Q&A tests'):
                by_answer = {r['answer_id']:r for r in history}
                saved = st.selectbox('Saved answer', list(by_answer), key='qa-history',
                    format_func=lambda key:(by_answer[key].get('gold_query_id') or 'Free question')+' · '+by_answer[key]['created_at'])
                if st.button('Open saved answer', key='qa-open'):
                    st.session_state['qa-answer-id'] = saved
                    st.rerun()
                st.download_button('Download Q&A results', json.dumps(history,indent=2).encode(),
                    file_name='BLACKWING-QA-results.json', mime='application/json', key='qa-export')
        answer_id = st.session_state.get('qa-answer-id')
        if not answer_id:
            return
        answer = copilot.request('GET', '/api/v1/qa/answers/'+answer_id)
        with st.chat_message('user'):
            st.text(answer['question'])
        with st.chat_message('assistant'):
            st.text(answer['answer'])
            st.caption(answer['notice'])
            if answer['abstained']:
                st.info('The answer abstains. Check whether that uncertainty is justified by the available evidence.')
            if not answer['citations']:
                st.warning('No retrieved passage was cited. This answer has no verified source coverage.')
            for source in answer['sources']:
                with st.expander(source['passage_id']):
                    st.caption(f"{source.get('source_id')} · page {source.get('page')}")
                    st.text(source['excerpt'])
            if answer['missing_information']:
                st.write(answer['missing_information'])
            if answer['limitations']:
                st.write(answer['limitations'])
        if answer.get('gold_query_id'):
            if st.button('Compare with gold evaluation', key='qa-reveal'):
                st.session_state['qa-reference-for'] = answer_id
            if st.session_state.get('qa-reference-for') != answer_id:
                return
            reference = copilot.request('GET', '/api/v1/qa/answers/'+answer_id+'/reference')
            generated, gold = st.columns(2)
            generated.subheader('BLACKWING answer')
            generated.text(answer['answer'])
            gold.subheader('Gold evaluation answer')
            gold.text(reference['expected_answer'])
            gold.caption(str(reference.get('source_ids') or '')+' · '+str(reference.get('locator') or ''))
            st.info(reference['notice'])
        else:
            st.info('No unique approved answer matches this question in the existing gold evaluation workbook. Verify its cited sources and record a human verdict.')
        with st.form('qa-review-form'):
            verdict = st.selectbox('Human verdict', ['unassessed','supported','partly_supported','incorrect'], key='qa-verdict')
            reviewer = st.text_input('Reviewer', key='qa-reviewer')
            rationale = st.text_area('Verification notes', key='qa-rationale')
            save = st.form_submit_button('Save verification', key='qa-save-review')
        if save:
            copilot.request('POST', '/api/v1/qa/answers/'+answer_id+'/review', json={
                'verdict':verdict,'reviewer':reviewer,'rationale':rationale})
            st.success('Human verification saved in backend SQLite.')
        if answer.get('review'):
            st.json(answer['review'])
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get('detail')
            message = detail.get('message') if isinstance(detail,dict) else detail
        except ValueError:
            message = None
        st.error(message or f'Q&A request failed with HTTP {exc.response.status_code}.')
    except httpx.HTTPError:
        st.error('Q&A connection interrupted. Check saved answers before retrying.')
