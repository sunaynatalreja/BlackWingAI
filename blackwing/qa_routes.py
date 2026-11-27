"""Versioned Q&A endpoints; evaluation never mutates investigation reports."""
from datetime import datetime, timezone
import logging

from fastapi import APIRouter, Depends, HTTPException
from blackwing.qa import QAAnswer, QARequest, QAReview, QAInputError, QAReferenceNotFound, answer_question, generate_answer, query_catalog, reference_for
from blackwing.rag.retriever import retriever
from blackwing.diagnostics import safe_error
from blackwing.context_guard import IrrelevantContext

logger = logging.getLogger('blackwing.qa')


def get_qa_generator():
    return generate_answer


def build_router(get_repository, get_settings):
    router = APIRouter(prefix='/api/v1/qa', tags=['Q&A'])

    @router.get('/queries', summary='List preserved gold questions without approved answers')
    def queries():
        try:
            return query_catalog()
        except (OSError, ValueError, KeyError, ImportError) as exc:
            raise HTTPException(503, detail='Gold question workbook unavailable; configure BLACKWING_GOLD_QUERY_FILE and install project requirements.') from exc


    @router.get('/answers', response_model=list[QAAnswer])
    def answers(repo=Depends(get_repository)):
        return repo.list_qa_answers()


    @router.get('/answers/{answer_id}', response_model=QAAnswer)
    def answer(answer_id:str, repo=Depends(get_repository)):
        row=repo.get_qa_answer(answer_id)
        if row is None:
            raise HTTPException(404, detail='Q&A answer not found.')
        return row


    @router.post('/answers', response_model=QAAnswer, status_code=201,
        summary='Retrieve aviation evidence and generate one answer',
        description='Rejects harmful instructions and questions outside aviation safety context with HTTP 422 before retrieval or model inference.',
        responses={422: {'description': 'Input is not a relevant aviation safety context.'}})
    async def create_answer(request:QARequest, settings=Depends(get_settings), repo=Depends(get_repository),
                            generate=Depends(get_qa_generator)):
        if repo.list_cases(status='analysis_running', limit=1, offset=0)[0]:
            raise HTTPException(409, detail='Wait for the active investigation to finish before starting Q&A.')
        try:
            return await answer_question(request, settings=settings, repo=repo, retriever=retriever, generate=generate)
        except IrrelevantContext as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        except QAReferenceNotFound as exc:
            raise HTTPException(404, detail='Selected question or case was not found.') from exc
        except QAInputError as exc:
            raise HTTPException(422, detail='Use the unchanged selected gold query, or submit a free question separately.') from exc
        except (OSError, ImportError) as exc:
            raise HTTPException(503, detail='Q&A configuration unavailable. Check the existing index, question workbook and project requirements.') from exc
        except Exception as exc:
            diagnostic=safe_error(exc)
            logger.error('Q&A failed: code=%s error_type=%s', diagnostic['code'], diagnostic['type'])
            raise HTTPException(502, detail={'code':diagnostic['code'], 'message':
                'Q&A did not complete. No successful answer or correctness score was recorded. Check backend diagnostics before retrying.'}) from exc


    @router.get('/answers/{answer_id}/reference', summary='Reveal gold answer after the generated answer has been saved')
    def reference(answer_id:str, repo=Depends(get_repository)):
        row=answer(answer_id,repo)
        if not row.get('gold_query_id'):
            raise HTTPException(404, detail='This free question has no matching gold reference.')
        try:
            return reference_for(row)
        except ValueError as exc:
            raise HTTPException(409, detail='Gold workbook changed after generation; comparison refused.') from exc
        except (KeyError,OSError,ImportError) as exc:
            raise HTTPException(503, detail='The preserved reference is unavailable.') from exc


    @router.post('/answers/{answer_id}/review', response_model=QAAnswer)
    def review(answer_id:str, request:QAReview, repo=Depends(get_repository)):
        row=answer(answer_id,repo)
        # Validate workbook identity before attaching a gold-comparison verdict.
        if row.get('gold_query_id'):
            reference(answer_id,repo)
        payload=request.model_dump(mode='json')
        history=[*row.get('review_history',[]), {**payload,'created_at':datetime.now(timezone.utc).isoformat()}]
        updated={**row,'review':payload,'review_history':history}
        repo.save_qa_answer(answer_id,updated)
        return updated

    return router
