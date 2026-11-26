"""Explicit final-report evaluation endpoints for evaluator-only gold PDFs."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from blackwing.evaluation.final_report import (
    GoldReferenceUnavailable,
    evaluate_final_report,
    load_reference_for_case,
)

PROCESS_ID = str(uuid4())
_locks: dict[str, asyncio.Lock] = {}
logger = logging.getLogger("blackwing.final_report_evaluation_routes")


def _context(repo, case_id: str, run_id: str):
    case = repo.get_case(case_id)
    run = repo.get_run(run_id)
    if not case or not run or run.get("case_id") != case_id:
        raise HTTPException(404, detail="Investigation run not found.")
    if run.get("status") != "completed" or not run.get("result"):
        raise HTTPException(409, detail="A completed saved final report is required for evaluation.")
    return case, run


def _recover(repo, run):
    current = deepcopy(run.get("final_report_evaluation") or {})
    # Older evaluations marked themselves completed even when some sections
    # only had semantic metrics and their LLM judge response was malformed.
    # Surface those as retryable instead of silently treating them as done.
    if current.get("status") == "completed":
        sections = current.get("sections") or {}
        if any(not row.get("llm_judge") for row in sections.values()):
            current.update(status="incomplete")
            current.setdefault("events", []).append({
                "event": "missing_judge_scores_detected",
                "at": datetime.now(timezone.utc).isoformat(),
                "reason": "One or more report sections have no LLM judge score and can be retried.",
            })
            repo.patch_run(run["run_id"], {"final_report_evaluation": current})
    if current.get("status") in {"queued", "running"} and current.get("process_id") != PROCESS_ID:
        current.update(status="interrupted")
        current.setdefault("events", []).append({
            "event": "execution_interrupted", "at": datetime.now(timezone.utc).isoformat(),
            "reason": "Evaluation process restarted; completed section metrics are retained.",
        })
        repo.patch_run(run["run_id"], {"final_report_evaluation": current})
    return current


def build_router(get_repository, get_settings):
    router = APIRouter(prefix="/api/v1/cases", tags=["final report evaluation"])

    @router.get("/{case_id}/runs/{run_id}/final-report-evaluation",
                summary="Get final-report evaluation status",
                description="Returns evaluator-only reference matching, per-section judge/semantic results, warnings, and progress for a completed report. This read operation does not start model evaluation. Gold content is not used for generation or RAG.",
                responses={404: {"description": "Case or run not found"},
                           409: {"description": "A completed report is required"}})
    def status(case_id: str, run_id: str, repo=Depends(get_repository), settings=Depends(get_settings)):
        case, run = _context(repo, case_id, run_id)
        saved = _recover(repo, run)
        if saved:
            return saved
        base = {
            "label": "Final report section evaluation", "run_id": run_id,
            "human_review_required": True,
            "notice": "Evaluator-only reference. This PDF is excluded from generation and RAG.",
        }
        if not settings.final_report_gold_dir:
            return {**base, "status": "unavailable",
                    "reason": "Set BLACKWING_FINAL_REPORT_GOLD_DIR to the evaluator-only gold-report directory."}
        try:
            reference = load_reference_for_case(settings.final_report_gold_dir, case, run["result"])
        except GoldReferenceUnavailable as exc:
            return {**base, "status": "not_assessable", "reason": str(exc)}
        except Exception:
            return {**base, "status": "unavailable",
                    "reason": "The matching gold PDF could not be read. Check its format and access permissions."}
        return {**base, "status": "not_requested", "reference_id": reference["reference_id"],
                "reference_source": reference["source_name"], "reference_page_count": reference["page_count"],
                "section_mappings": reference["section_mappings"],
                "metrics": ["LLM-as-a-judge", "BGE cosine similarity/distance", "BERTScore"]}

    @router.post("/{case_id}/runs/{run_id}/final-report-evaluation", status_code=202,
                 summary="Start or retry final-report evaluation",
                 description="Queues section-level LLM-as-a-Judge, BGE cosine, and BERTScore comparison against the uniquely matched evaluator-only gold PDF. Existing validated section results are retained on retry. This operation may use hosted inference and embedding quotas.",
                 responses={202: {"description": "Evaluation queued or already running"},
                            404: {"description": "Case or run not found"},
                            409: {"description": "A completed report is required, a run is active, or no unique matching gold PDF exists"},
                            503: {"description": "Gold directory or evaluation service is unavailable"}})
    async def start(case_id: str, run_id: str, background: BackgroundTasks,
                    repo=Depends(get_repository), settings=Depends(get_settings)):
        async with _locks.setdefault(run_id, asyncio.Lock()):
            case, run = _context(repo, case_id, run_id)
            if not settings.final_report_gold_dir:
                raise HTTPException(503, detail="Set BLACKWING_FINAL_REPORT_GOLD_DIR to the evaluator-only gold-report directory.")
            active, _ = repo.list_cases(status="analysis_running", limit=1, offset=0)
            if active:
                raise HTTPException(409, detail="Wait for the active investigation to finish before starting evaluation.")
            prior = _recover(repo, run)
            if prior.get("status") in {"queued", "running"}:
                return prior
            try:
                reference = await asyncio.to_thread(load_reference_for_case,
                    settings.final_report_gold_dir, case, run["result"])
            except GoldReferenceUnavailable as exc:
                raise HTTPException(409, detail=str(exc)) from None
            except Exception as exc:
                logger.warning("Final report reference unavailable; error_type=%s", type(exc).__name__)
                raise HTTPException(503, detail="The configured evaluator PDF could not be read or its report sections could not be extracted.") from None
            queued = {
                **deepcopy(prior),
                "status": "queued", "label": "Final report section evaluation", "run_id": run_id,
                "reference_id": reference["reference_id"], "reference_source": reference["source_name"],
                "reference_sha256": reference["reference_sha256"], "reference_page_count": reference["page_count"],
                "section_mappings": {**reference["section_mappings"], **(prior.get("section_mappings") or {})},
                "sections_total": len((run["result"].get("generated_report") or {}).get("sections", [])),
                "sections_completed": len(prior.get("sections") or {}),
                "sections": deepcopy(prior.get("sections") or {}), "human_review_required": True,
                "notice": "Evaluator-only reference. This PDF is excluded from generation and RAG.",
                "process_id": PROCESS_ID,
            }
            repo.patch_run(run_id, {"final_report_evaluation": queued})

            def save(value):
                value["process_id"] = PROCESS_ID
                repo.patch_run(run_id, {"final_report_evaluation": value})

            async def execute():
                try:
                    await evaluate_final_report(deepcopy(run["result"]), reference, settings,
                                                save=save, run_id=run_id, prior=prior)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    if isinstance(exc, (KeyboardInterrupt, GeneratorExit)):
                        raise
                    current = deepcopy((repo.get_run(run_id) or {}).get("final_report_evaluation") or queued)
                    current.update(status="failed", error_code="final_report_evaluation_failed",
                                   error_type=type(exc).__name__)
                    try:
                        save(current)
                    except Exception:
                        logger.error("Final report evaluation failure could not be persisted; run=%s", run_id)

            background.add_task(execute)
            return queued

    return router
