"""LangGraph multi-agent workflow backed by a configurable chat model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

import httpx
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from blackwing.agents.base import AgentExecutionError, AgentOutputValidationError, AgentSpec
from blackwing.agents.compliance.agent import unassessed_output as compliance_unassessed_output
from blackwing.agents.compliance.evidence import ground_compliance_output
from blackwing.agents.risk.evidence import ground_risk_output
from blackwing.agents.similar_case.agent import build_output as similar_case_output
from blackwing.agents.investigation.grounding import ground_investigation_output
from blackwing.agents.incident_identity import conflicts_with_case
from blackwing.agents.mitigation.grounding import ground_mitigation_output
from blackwing.agents.critic.grounding import audit_critic_output
from blackwing.agents.verification.grounding import ground_verification_output
from blackwing.agents.report.workflow import create_report
from blackwing.agents.registry import AGENT_SPECS
from blackwing.agents.runtime import invoke_agent
from blackwing.config import Settings

logger = logging.getLogger("blackwing.workflow")


class CaseState(TypedDict, total=False):
    case_id: str
    narrative: str
    jurisdiction: str | None
    incident_date: str | None
    source_incident_id: str | None
    excluded_evidence: list[str]
    evidence: list[dict[str, Any]]
    agent_evidence: dict[str, list[dict[str, Any]]] | None
    outputs: dict[str, dict[str, Any]]
    trace: list[dict[str, Any]]


def _agent_node(spec: AgentSpec, client: httpx.AsyncClient, settings: Settings):
    async def run(state: CaseState) -> dict[str, Any]:
        role = spec.key
        fallback_used = False
        report_trace: list[dict[str, Any]] = []
        try:
            # Opt-in RAG provides role-specific sources; legacy runs use shared evidence.
            agent_packets = state.get("agent_evidence")
            evidence = (agent_packets.get(role, []) if agent_packets is not None
                        else state.get("evidence", []))
            if role == "report":
                evidence = state.get("evidence", [])
            compliance_sources = {
                "regulation", "procedure", "operational_order",
            }
            if role == "report":
                output, report_trace = await create_report(
                    client=client,
                    settings=settings,
                    narrative=state["narrative"],
                    evidence=evidence,
                    prior_outputs=state.get("outputs", {}),
                )
                fallback_used = any(event.get("status") == "deterministic_fallback" for event in report_trace)
            elif role == "similar_case":
                output = similar_case_output(state["narrative"], evidence,
                                             current_incident_id=state.get("source_incident_id"))
            elif role == "compliance" and not any(
                str(item.get("document_type", "")).casefold() in compliance_sources
                or str(item.get("source_type", "")).casefold() in compliance_sources
                for item in evidence
            ):
                # With no authoritative rule text, the model has no compliance
                # source to assess. Return an explicit, schema-valid unassessed result.
                output = compliance_unassessed_output()
            else:
                try:
                    output = await invoke_agent(
                        spec,
                        client=client,
                        settings=settings,
                        narrative=state["narrative"],
                        evidence=evidence,
                        prior_outputs=state.get("outputs", {}),
                    )
                except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                    if role not in {"critic", "verification"}:
                        raise
                    # Both roles have deterministic cross-agent grounding checks.
                    # If their optional model pass cannot fit the context or fails
                    # output validation, preserve a complete report and disclose
                    # that the model pass was skipped.
                    context_budget_exceeded = "token_budget" in str(exc).casefold()
                    if role == "critic":
                        from blackwing.agents.critic.schema import CriticOutput
                        logger.warning(
                            "Critic model review unavailable; deterministic audit only; reason=%s "
                            "error_type=%s detail=%s",
                            "context_budget_exceeded" if context_budget_exceeded else "invalid_model_response",
                            type(exc).__name__, str(exc)[:500],
                        )
                        output = CriticOutput(
                            issues=[{
                                "issue_type": "critic_model_context_budget_exceeded" if context_budget_exceeded else "critic_model_output_invalid",
                                "severity": "review",
                                "affected_output": "critic",
                                "description": (
                                    "The Critic input exceeded the safe model context budget, so no Critic model response was produced; deterministic cross-agent checks were applied."
                                    if context_budget_exceeded else
                                    "The Critic model response did not match its response schema; deterministic cross-agent checks were applied instead."
                                ),
                                "suggested_correction": "Review the deterministic audit and service log; reduce the Critic context or correct the response schema issue before retrying.",
                            }],
                            missing_analysis=[
                                "Critic model input exceeded the safe context budget; only deterministic audit checks ran."
                                if context_budget_exceeded else
                                "Model-generated adversarial review was unavailable; only deterministic audit checks ran."
                            ],
                            overall_assessment="requires_human_review",
                        ).model_dump(mode="json")
                    else:
                        from blackwing.agents.verification.schema import VerificationOutput
                        logger.warning(
                            "Verification model review unavailable; deterministic checks only; reason=%s "
                            "error_type=%s detail=%s",
                            "context_budget_exceeded" if context_budget_exceeded else "invalid_model_response",
                            type(exc).__name__, str(exc)[:500],
                        )
                        output = VerificationOutput(
                            claims_needing_source=[
                                (
                                    "Verification input exceeded the safe model context budget; deterministic provenance and completeness checks were applied."
                                    if context_budget_exceeded else
                                    "Model-generated verification was unavailable; deterministic provenance and completeness checks were applied."
                                )
                            ],
                            completeness_status="incomplete",
                            human_approval_required=True,
                        ).model_dump(mode="json")
                    fallback_used = True
                if role == "investigation":
                    output = ground_investigation_output(
                        output, evidence, incident_id=state.get("source_incident_id")
                    )
                    if state.get("excluded_evidence"):
                        output["incident_summary"] = "The submitted incident narrative describes: " + state["narrative"].strip()
                        output["unknowns"].append(
                            "Evidence with conflicting incident identifiers was excluded; verify source identity and obtain current-incident records."
                        )
                elif role == "compliance":
                    output = ground_compliance_output(
                        output, evidence,
                        jurisdiction=state.get("jurisdiction"),
                        incident_date=state.get("incident_date"),
                    )
                elif role == "risk":
                    output = ground_risk_output(output, evidence)
                elif role == "mitigation":
                    output = ground_mitigation_output(output, state.get("outputs", {}))
                elif role == "critic":
                    output = audit_critic_output(output, state.get("outputs", {}))
                elif role == "verification":
                    output = ground_verification_output(output, state.get("outputs", {}))
        except httpx.TimeoutException as exc:
            raise AgentExecutionError(role, "model_timeout", "The configured model provider did not respond before the configured timeout.") from exc
        except httpx.HTTPStatusError as exc:
            raise AgentExecutionError(role, "model_http_error", f"The configured model provider returned HTTP {exc.response.status_code}. Check provider, model, and API key settings.") from exc
        except httpx.RequestError as exc:
            raise AgentExecutionError(role, "model_unreachable", "The API could not connect to the configured model provider.") from exc
        except AgentOutputValidationError as exc:
            debug_detail = (
                f"validation_details={exc.diagnostic} response_preview={exc.response_preview}"
            )
            raise AgentExecutionError(
                role, "invalid_agent_output",
                "The model response did not match the agent's required JSON schema.",
                debug_detail=debug_detail,
            ) from exc
        except (json.JSONDecodeError, ValidationError) as exc:
            if isinstance(exc, ValidationError):
                detail = json.dumps(exc.errors(include_input=False, include_url=False), ensure_ascii=False)
            else:
                detail = json.dumps({"message": exc.msg, "line": exc.lineno, "column": exc.colno})
            raise AgentExecutionError(
                role, "invalid_agent_output",
                "The model response did not match the agent's required JSON schema.",
                debug_detail=f"error_type={type(exc).__name__} details={detail}",
            ) from exc
        except ValueError as exc:
            if "token_budget" in str(exc).casefold():
                raise AgentExecutionError(
                    role,
                    "context_budget_exceeded",
                    "The agent input exceeded the safe model context budget; no model response was produced.",
                ) from exc
            if "token preflight unavailable" in str(exc).casefold():
                raise AgentExecutionError(
                    role,
                    "tokenizer_unavailable",
                    "The required local Qwen tokenizer is unavailable; cache the configured tokenizer before starting the API.",
                    debug_detail=f"error_type={type(exc).__name__} detail={str(exc)[:500]}",
                ) from exc
            raise AgentExecutionError(
                role, "invalid_agent_output",
                "The model response did not match the agent's required JSON schema.",
                debug_detail=f"error_type={type(exc).__name__} detail={str(exc)[:1000]}",
            ) from exc
        except Exception as exc:
            raise AgentExecutionError(role, "agent_processing_error", "The agent failed while processing the model response.") from exc
        outputs = dict(state.get("outputs", {}))
        outputs[role] = output
        trace = [
            *state.get("trace", []),
            *report_trace,
            {
                "step": role,
                "agent": spec.name,
                "model": "deterministic-corpus-comparison" if role == "similar_case" else settings.model_name,
                "status": "deterministic_fallback" if fallback_used else "complete",
            },
        ]
        return {"outputs": outputs, "trace": trace}

    return run


def _build_graph(client: httpx.AsyncClient, settings: Settings):
    graph = StateGraph(CaseState)
    ordered_nodes = [spec.key for spec in AGENT_SPECS]
    for spec in AGENT_SPECS:
        graph.add_node(spec.key, _agent_node(spec, client, settings))
    graph.add_edge(START, ordered_nodes[0])
    for current, following in zip(ordered_nodes, ordered_nodes[1:]):
        graph.add_edge(current, following)
    graph.add_edge(ordered_nodes[-1], END)
    return graph.compile()


def _case_narrative_evidence(case_id: str, narrative: str) -> dict[str, Any]:
    """Give the submitted account a stable, case-scoped citation identity.

    This is explicitly an unverified user report. It is passed only through the
    current run and is never added to the shared retrieval corpus.
    """
    return {
        "evidence_id": f"CASE-NARRATIVE:{case_id}",
        "source_id": f"CASE-NARRATIVE:{case_id}",
        "source_type": "user_submitted",
        "evidence_role": "reported",
        "evidence_status": "USER_REPORTED_UNVERIFIED",
        "title": "Submitted incident narrative",
        "excerpt": narrative.strip(),
        "incident_id": case_id,
    }


async def analyze_incident(
    narrative: str,
    settings: Settings,
    *,
    case_id: str | None = None,
    evidence: list[dict[str, Any]] | None = None,
    retrieval_count: int = 0,
    agent_evidence: dict[str, list[dict[str, Any]]] | None = None,
    jurisdiction: str | None = None,
    incident_date: str | None = None,
    source_incident_id: str | None = None,
) -> dict[str, Any]:
    case_id = case_id or str(uuid4())
    # Apply the same boundary to shared and role-specific packets before any
    # model sees them. A conflicting report cannot establish current facts.
    all_rows = [*(evidence or []), *(row for rows in (agent_evidence or {}).values() for row in rows)]
    def source_key(row: dict[str, Any]) -> str:
        return str(row.get("source_id") or row.get("document_id") or row.get("evidence_id") or "unknown source")

    excluded = {source_key(row) for row in all_rows
                if str(row.get("source_type") or "").casefold() in {"official_factual", "official_preliminary", "user_submitted", "user_report"}
                and str(row.get("evidence_role") or "").casefold() not in {"historical", "historical_analogy", "similar_case"}
                and conflicts_with_case(row, narrative, incident_date)}
    def filter_conflicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [row for row in rows if source_key(row) not in excluded]

    evidence = filter_conflicts(evidence or [])
    if agent_evidence is not None:
        agent_evidence = {key: filter_conflicts(rows) for key, rows in agent_evidence.items()}
    # The source incident reference is different from the application's UUID.
    # If older callers omit it, recover from a scoped official case-factual chunk.
    if not source_incident_id:
        source_incident_id = next((
            str(row["incident_id"])
            for row in ((agent_evidence or {}).get("investigation") or [])
            if row.get("incident_id") and str(row.get("evidence_role") or "").casefold() == "case_factual"
        ), None)
    source_incident_id = source_incident_id or case_id
    narrative_evidence = _case_narrative_evidence(case_id, narrative)
    # The user's submitted account is reportable as an observation, not as an
    # independently verified finding. Keep it scoped to this analysis run.
    run_evidence = [narrative_evidence, *(evidence or [])]
    run_agent_evidence = None
    if agent_evidence is not None:
        run_agent_evidence = {key: list(rows) for key, rows in agent_evidence.items()}
        for role in ("investigation", "compliance", "risk", "mitigation", "critic", "verification"):
            rows = run_agent_evidence.get(role, [])
            # Put user narrative first so it survives each agent's bounded packet.
            run_agent_evidence[role] = [narrative_evidence, *[
                row for row in rows
                if str(row.get("evidence_id") or row.get("chunk_id")) != narrative_evidence["evidence_id"]
            ]]
    initial_state: CaseState = {
        "case_id": case_id,
        "narrative": narrative.strip(),
        "jurisdiction": jurisdiction,
        "incident_date": incident_date,
        "source_incident_id": source_incident_id,
        "excluded_evidence": sorted(excluded),
        "evidence": run_evidence,
        "agent_evidence": run_agent_evidence,
        "outputs": {},
        "trace": [
            {"step": "case_created", "status": "complete"},
            {
                "step": "source_retrieval",
                "status": "complete" if retrieval_count else "no_matches",
                "retrieved_document_count": retrieval_count,
            },
            {"step": "incident_identity_check", "status": "conflicts_excluded" if excluded else "no_explicit_conflicts",
             "excluded_source_ids": sorted(excluded)},
        ],
    }
    # Hosted IIT TLS may use a trusted local CA bundle. Never disable verification.
    verify = str(Path(settings.api_ca_cert).expanduser()) if settings.api_ca_cert else True
    async with httpx.AsyncClient(timeout=settings.model_timeout_seconds, verify=verify) as client:
        graph = _build_graph(client, settings)
        final_state = await graph.ainvoke(initial_state)

    outputs = dict(final_state["outputs"])
    generated_report = outputs.pop("report", None)
    risk_score = outputs.get("risk", {}).get("numeric_score")
    trace = [
        *final_state["trace"],
        {
            "step": "provisional_5x5_risk_scoring",
            "status": "complete" if risk_score is not None else "incomplete",
            "detail": (
                f"score={risk_score}/25 rating={outputs['risk'].get('risk_rating')}"
                if risk_score is not None else "Severity and likelihood ratings are incomplete."
            ),
        },
        {"step": "human_review", "status": "required"},
    ]
    return {
        "case_id": case_id,
        "status": "PRELIMINARY_REQUIRES_HUMAN_REVIEW",
        "workflow": " -> ".join(spec.key for spec in AGENT_SPECS),
        "model": settings.model_name,
        "agent_outputs": outputs,
        "generated_report": generated_report,
        "trace": trace,
        "limitations": [
            "Model-backed agent roles share the configured model; role separation is prompt and workflow separation, not separate model deployments.",
            "Historical retrieved cases are analogies only; their findings do not establish the cause of the current incident.",
            "BGE dense retrieval falls back to BM25 when local embedding weights are not available.",
            "Risk scoring is not configured; the Risk Agent must not return a numeric score.",
            "Human review and approval are required before any result is treated as a finding or action.",
        ],
    }
