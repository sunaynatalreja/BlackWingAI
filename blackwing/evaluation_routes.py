"""Explicit post-generation evaluation jobs; GET never invokes a model."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from blackwing.evaluation.materials import MaterialsUnavailable, case_materials, digest
from blackwing.evaluation.report_judge import evaluate_report, evaluation_identity

PROCESS_ID = str(uuid4())
_locks = {}
logger = logging.getLogger('blackwing.report_evaluation')


def get_question_judge():
    from blackwing.evaluation.report_judge import judge_question
    return judge_question


def _context(repo, case_id, run_id):
    case = repo.get_case(case_id)
    run = repo.get_run(run_id)
    if not case or not run or run.get('case_id') != case_id:
        raise HTTPException(404, detail='Investigation run not found.')
    if run.get('status') != 'completed' or not run.get('result'):
        raise HTTPException(409, detail='A completed saved report is required for question evaluation.')
    return case, run


def _recover(repo, run):
    cached = deepcopy(run.get('question_evaluation') or {})
    if cached.get('status') in {'queued', 'running'} and cached.get('process_id') != PROCESS_ID:
        cached.update(status='interrupted', score=None)
        cached.setdefault('events', []).append({'event': 'execution_interrupted',
            'at': datetime.now(timezone.utc).isoformat(),
            'reason': 'Evaluation process changed; validated question results are retained.'})
        repo.patch_run(run['run_id'], {'question_evaluation': cached})
    return cached


def build_router(get_repository, get_settings):
    router = APIRouter(prefix='/api/v1/cases', tags=['report evaluation'])

    @router.get('/{case_id}/runs/{run_id}/report-evaluation')
    def status(case_id: str, run_id: str, repo=Depends(get_repository)):
        case, run = _context(repo, case_id, run_id)
        cached = _recover(repo, run)
        if cached:
            return cached
        base = {'score': None, 'label': 'Gold comparison score',
                'comparison_mode': 'saved_report_questions', 'human_review_required': True}
        try:
            materials = case_materials(case)
        except MaterialsUnavailable as exc:
            # Preserve the old claim/safety-policy comparator where approved
            # question coverage is unavailable. Never label it question coverage.
            return {**base, 'status': 'unavailable', 'reason': exc.reason,
                    'reference_case': exc.reference_case, 'legacy_fallback': True}
        except Exception:
            return {**base, 'status': 'unavailable', 'legacy_fallback': False,
                    'reason': 'Matching evaluation materials could not be verified. Check the protected workbook, reference files and configuration.'}
        return {**base, 'status': 'not_requested', 'reference_case': materials.get('reference_case', True),
                'evaluation_split': materials.get('evaluation_split'),
                'reference_id': materials['reference_id'], 'questions_total': len(materials['questions'])}

    @router.post('/{case_id}/runs/{run_id}/report-evaluation', status_code=202)
    async def start(case_id: str, run_id: str, background: BackgroundTasks,
                    repo=Depends(get_repository), settings=Depends(get_settings), judge=Depends(get_question_judge)):
        async with _locks.setdefault(run_id, asyncio.Lock()):
            case, run = _context(repo, case_id, run_id)
            active, _ = repo.list_cases(status='analysis_running', limit=1, offset=0)
            if active:
                raise HTTPException(409, detail='Wait for the active investigation before starting evaluation.')
            prior = _recover(repo, run)
            if prior.get('status') in {'queued', 'running'}:
                return prior
            try:
                materials = await asyncio.to_thread(case_materials, case)
            except Exception:
                raise HTTPException(503, detail='Matching approved evaluation questions/reference unavailable; check protected workbook and reference configuration.') from None
            report = deepcopy(run['result'])
            snapshot = deepcopy(run.get('evaluation_input_snapshot') or {})
            identity = evaluation_identity(report, materials, settings, run_id=run_id, snapshot=snapshot)
            if prior.get('status') == 'completed' and prior.get('cache_key') == digest(identity):
                return prior
            reusable = prior.get('cache_key') == digest(identity)
            retained = [r for r in prior.get('judgments', []) if not r.get('validation_error')] if reusable else []
            queued = {'status': 'queued', 'score': None, 'label': 'Gold comparison score',
                      'comparison_mode': 'saved_report_questions', 'process_id': PROCESS_ID,
                      'reference_id': materials['reference_id'], 'reference_case': materials.get('reference_case', True),
                      'evaluation_split': materials.get('evaluation_split'),
                      'human_review_required': True,
                      'run_id': run_id, 'questions_total': len(materials['questions']),
                      'questions_completed': len(retained), 'judgments': retained, 'events': []}
            repo.patch_run(run_id, {'question_evaluation': queued})
            def save(value):
                value['process_id'] = PROCESS_ID
                repo.patch_run(run_id, {'question_evaluation': value})
            async def execute():
                try:
                    await evaluate_report(report, materials, settings, run_id=run_id,
                        snapshot=snapshot, prior=prior, save=save, judge=judge)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    if isinstance(exc, (KeyboardInterrupt, GeneratorExit)):
                        raise
                    current = deepcopy((repo.get_run(run_id) or {}).get('question_evaluation') or queued)
                    current.update(status='failed', score=None, error_code='report_evaluation_failed')
                    try:
                        save(current)
                    except Exception:
                        logger.error('Evaluation failure could not be persisted; run=%s', run_id)
            background.add_task(execute)
            return queued
    return router
