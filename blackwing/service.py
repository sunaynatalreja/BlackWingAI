"""LangGraph multi-agent workflow backed by a configurable chat model."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, TypedDict
from uuid import uuid4

import httpx
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from blackwing.agents.base import AgentExecutionError, AgentSpec
from blackwing.agents.compliance.agent import unassessed_output as compliance_unassessed_output
from blackwing.agents.compliance.evidence import ground_compliance_output
from blackwing.agents.risk.evidence import ground_risk_output
from blackwing.agents.investigation.grounding import ground_investigation_output
from blackwing.agents.mitigation.grounding import ground_mitigation_output
from blackwing.agents.critic.grounding import audit_critic_output
from blackwing.agents.verification.grounding import ground_verification_output
from blackwing.agents.report.workflow import create_report
from blackwing.agents.registry import AGENT_SPECS
from blackwing.agents.runtime import invoke_agent
from blackwing.config import Settings
from blackwing.diagnostics import execution_audit, agent_execution, emit, safe_error, MESSAGES
from blackwing.tools import tool_audit, find_similar_incidents
from blackwing.planning import plan_investigation, correction_target, review_correction, MAX_EXECUTION_STEPS

logger = logging.getLogger("blackwing.workflow")


class CaseState(TypedDict, total=False):
    case_id: str
    narrative: str
    jurisdiction: str | None
    incident_date: str | None
    source_incident_id: str | None
    evidence: list[dict[str, Any]]
    agent_evidence: dict[str, list[dict[str, Any]]] | None
    outputs: dict[str, dict[str, Any]]
    trace: list[dict[str, Any]]
    plan: dict[str, Any]
    execution_steps: int
    correction_attempts: int
    correction_context: dict[str, Any]


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
                    deterministic=state.get("plan", {}).get("task_modes", {}).get("report") == "deterministic_assembly",
                )
                fallback_used = any(event.get("status") == "deterministic_fallback" for event in report_trace)
            elif role == "similar_case":
                output = find_similar_incidents(narrative=state["narrative"], evidence=evidence,
                                             current_incident_id=state.get("source_incident_id"))
            elif role == "compliance" and (state.get("plan", {}).get("task_modes", {}).get("compliance") == "unassessed" or not any(
                str(item.get("document_type", "")).casefold() in compliance_sources
                or str(item.get("source_type", "")).casefold() in compliance_sources
                for item in evidence
            )):
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
                        prior_outputs={**state.get("outputs", {}), **({"correction_request": state["correction_context"]} if state.get("correction_context") else {})},
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
                            type(exc).__name__, safe_error(exc),
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
                            type(exc).__name__, safe_error(exc),
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
                    emit("model_review_unavailable", error=safe_error(exc, "context_budget_exceeded" if context_budget_exceeded else "invalid_agent_output"))
                    fallback_used = True
                if role == "investigation":
                    output = ground_investigation_output(
                        output, evidence, incident_id=state.get("source_incident_id")
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
        except ValueError as exc:
            if "token_budget" in str(exc).casefold():
                raise AgentExecutionError(
                    role,
                    "context_budget_exceeded",
                    "The agent input exceeded the safe model context budget; no model response was produced.",
                ) from exc
            raise AgentExecutionError(role, "invalid_agent_output", "The model response did not match the agent's required JSON schema.") from exc
        except (json.JSONDecodeError, ValidationError) as exc:
            raise AgentExecutionError(role, "invalid_agent_output", "The model response did not match the agent's required JSON schema.") from exc
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


def _build_graph(client: httpx.AsyncClient, settings: Settings, on_update=None):
    graph = StateGraph(CaseState)
    specs = {spec.key: spec for spec in AGENT_SPECS}

    def publish(state):
        if on_update:
            on_update(state)

    def planner(state):
        plan = plan_investigation(state)
        result = {"plan": plan, "execution_steps": 1, "correction_attempts": 0,
                  "trace": [*state.get("trace", []), {"step": "planner", "status": "complete", "detail": plan}]}
        publish({**state, **result})
        return result

    def executor(role):
        async def run(state):
            steps = state.get("execution_steps", 0)
            if steps >= MAX_EXECUTION_STEPS:
                raise AgentExecutionError(role, "execution_step_limit", "Investigation exceeded its execution budget; human review required.")
            with agent_execution(specs[role], settings, state.get("plan", {}).get("task_modes", {}).get(role)):
                try:
                    result = await _agent_node(specs[role], client, settings)(state)
                except SystemExit as exc:
                    # Contain an exit attempt inside the graph task itself, before
                    # asyncio can treat a BaseException as process termination.
                    raise AgentExecutionError(role, "process_exit_attempt", MESSAGES["process_exit_attempt"]) from exc
            result["execution_steps"] = steps + 1
            if role == "critic":
                target = correction_target({**state, **result})
                result["trace"].append({"step": "correction_decision", "status": "target_selected" if target else "human_review",
                                        "detail": {"target": target, "reason": "existing evidence supports one correction" if target else "no supported correctable issue; retain unresolved findings"}})
            publish({**state, **result})
            return result
        return run

    async def correct(state):
        target = correction_target(state)
        if not target:
            return {}
        before = state["outputs"]["critic"]
        trigger = {"step": "targeted_correction", "status": "started", "detail": {"target": target, "issues": before["issues"]}}
        revised = {**state, "trace": [*state["trace"], trigger], "correction_attempts": 1,
                   "correction_context": {"target": target, "issues": before["issues"],
                                          "instruction": "Correct only this component using existing sources; preserve unknowns. Never fabricate evidence."}}
        publish(revised)
        try:
            update = await executor(target)(revised)
            revised.update(update)
            revised["outputs"]["mitigation"] = ground_mitigation_output(revised["outputs"].get("mitigation", {}), revised["outputs"])
            critic, resolved = review_correction(before, revised["outputs"])
            revised["outputs"] = {**revised["outputs"], "critic": critic}
            status = "corrected" if resolved else "unresolved_requires_human_review"
        except AgentExecutionError as exc:
            # Preserve the original valid outputs. One failed target is not a full restart.
            revised["execution_steps"] = state["execution_steps"] + 1
            resolved = []
            status = "failed_requires_human_review"
            revised["trace"].append({"step": "correction_error", "status": "failed", "detail": {"target": target, "error_code": exc.code}})
        revised["trace"] = [*revised["trace"], {"step": "correction_outcome", "status": status,
                             "detail": {"target": target, "resolved_issues": resolved, "human_review_required": True}}]
        revised["correction_context"] = {}
        publish(revised)
        return revised

    def route(state):
        return "correction" if correction_target(state) else "verification"

    graph.add_node("planner", planner)
    for spec in AGENT_SPECS:
        graph.add_node(spec.key, executor(spec.key))
    graph.add_node("correction", correct)
    ordered = [spec.key for spec in AGENT_SPECS]
    graph.add_edge(START, "planner")
    graph.add_edge("planner", ordered[0])
    for current, following in zip(ordered, ordered[1:]):
        if current != "critic":
            graph.add_edge(current, following)
    graph.add_conditional_edges("critic", route, {"correction": "correction", "verification": "verification"})
    graph.add_edge("correction", "verification")
    graph.add_edge(ordered[-1], END)
    return graph.compile()


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
    on_update=None,
    on_tool_event=None,
    on_execution_event=None,
) -> dict[str, Any]:
    case_id = case_id or str(uuid4())
    # The source incident reference is different from the application's UUID.
    # If older callers omit it, recover from a scoped official case-factual chunk.
    if not source_incident_id:
        source_incident_id = next((
            str(row["incident_id"])
            for row in ((agent_evidence or {}).get("investigation") or [])
            if row.get("incident_id") and str(row.get("evidence_role") or "").casefold() == "case_factual"
        ), None)
    source_incident_id = source_incident_id or case_id
    initial_state: CaseState = {
        "case_id": case_id,
        "narrative": narrative.strip(),
        "jurisdiction": jurisdiction,
        "incident_date": incident_date,
        "source_incident_id": source_incident_id,
        "evidence": evidence or [],
        "agent_evidence": agent_evidence,
        "outputs": {},
        "trace": [
            {"step": "case_created", "status": "complete"},
            {
                "step": "source_retrieval",
                "status": "complete" if retrieval_count else "no_matches",
                "retrieved_document_count": retrieval_count,
            },
        ],
    }
    # Hosted IIT TLS may use a trusted local CA bundle. Never disable verification.
    ca_bundle = os.environ.get("BLACKWING_LLM_CA_CERT")
    if ca_bundle and not os.path.isfile(ca_bundle):
        raise ValueError("BLACKWING_LLM_CA_CERT must point to an existing trusted CA bundle")
    with execution_audit(case_id, on_execution_event) as execution_events:
        emit("workflow_started", step="workflow_execution", detail={"phase": "initializing_model_client"})
        async with httpx.AsyncClient(timeout=settings.model_timeout_seconds,
                                     verify=ca_bundle if ca_bundle else True) as client:
            tool_events = []
            def record_tool(event):
                tool_events.append(event)
                if on_tool_event:
                    on_tool_event(event)
            graph = _build_graph(client, settings, on_update)
            emit("workflow_ready", step="workflow_execution")
            with tool_audit(case_id, record_tool):
                final_state = await graph.ainvoke(initial_state, {"recursion_limit": 16})

    outputs = dict(final_state["outputs"])
    generated_report = outputs.pop("report", None)
    # Retain the evidence-validated risk assessment from the Risk Agent.
    # Numeric score remains null; the FAA-reference index is categorical.
    trace = [
        *final_state["trace"],
        *tool_events,
        *execution_events,
        {"step": "investigation_outcome", "status": "requires_human_review", "detail": {
            "execution_steps": final_state["execution_steps"], "correction_attempts": final_state["correction_attempts"],
            "unresolved_issues": len(outputs.get("critic", {}).get("issues", []))}},
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
            "Risk classifications are preliminary reference-matrix assessments, not approved operational risk decisions; numeric_score remains null.",
            "Human review and approval are required before any result is treated as a finding or action.",
        ],
    }
