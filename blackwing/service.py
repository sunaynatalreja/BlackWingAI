"""LangGraph multi-agent workflow backed by a configurable chat model."""

from __future__ import annotations

import json
from typing import Any, TypedDict
from uuid import uuid4

import httpx
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from blackwing.agents.base import AgentExecutionError, AgentSpec
from blackwing.agents.compliance.agent import unassessed_output as compliance_unassessed_output
from blackwing.agents.compliance.evidence import ground_compliance_output
from blackwing.agents.risk.evidence import ground_risk_output
from blackwing.agents.similar_case.agent import build_output as similar_case_output
from blackwing.agents.investigation.agent import remove_historical_case_evidence_ids
from blackwing.agents.registry import AGENT_SPECS
from blackwing.agents.runtime import invoke_agent
from blackwing.config import Settings


class CaseState(TypedDict, total=False):
    case_id: str
    narrative: str
    jurisdiction: str | None
    incident_date: str | None
    evidence: list[dict[str, Any]]
    agent_evidence: dict[str, list[dict[str, Any]]] | None
    outputs: dict[str, dict[str, Any]]
    trace: list[dict[str, Any]]


def _agent_node(spec: AgentSpec, client: httpx.AsyncClient, settings: Settings):
    async def run(state: CaseState) -> dict[str, Any]:
        role = spec.key
        try:
            # Opt-in RAG provides role-specific sources; legacy runs use shared evidence.
            agent_packets = state.get("agent_evidence")
            evidence = (agent_packets.get(role, []) if agent_packets is not None
                        else state.get("evidence", []))
            compliance_sources = {
                "regulation", "procedure",
            }
            if role == "similar_case":
                output = similar_case_output(state["narrative"], evidence)
            elif role == "compliance" and not any(
                str(item.get("document_type", "")).casefold() in compliance_sources
                for item in evidence
            ):
                # With no authoritative rule text, the model has no compliance
                # source to assess. Return an explicit, schema-valid unassessed result.
                output = compliance_unassessed_output()
            else:
                output = await invoke_agent(
                    spec,
                    client=client,
                    settings=settings,
                    narrative=state["narrative"],
                    evidence=evidence,
                    prior_outputs=state.get("outputs", {}),
                )
                if role == "investigation":
                    output = remove_historical_case_evidence_ids(output, evidence)
                elif role == "compliance":
                    output = ground_compliance_output(
                        output, evidence,
                        jurisdiction=state.get("jurisdiction"),
                        incident_date=state.get("incident_date"),
                    )
                elif role == "risk":
                    output = ground_risk_output(output, evidence)
        except httpx.TimeoutException as exc:
            raise AgentExecutionError(role, "model_timeout", "The configured model provider did not respond before the configured timeout.") from exc
        except httpx.HTTPStatusError as exc:
            raise AgentExecutionError(role, "model_http_error", f"The configured model provider returned HTTP {exc.response.status_code}. Check provider, model, and API key settings.") from exc
        except httpx.RequestError as exc:
            raise AgentExecutionError(role, "model_unreachable", "The API could not connect to the configured model provider.") from exc
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise AgentExecutionError(role, "invalid_agent_output", "The model response did not match the agent's required JSON schema.") from exc
        except Exception as exc:
            raise AgentExecutionError(role, "agent_processing_error", "The agent failed while processing the model response.") from exc
        outputs = dict(state.get("outputs", {}))
        outputs[role] = output
        trace = [
            *state.get("trace", []),
            {
                "step": role,
                "agent": spec.name,
                "model": "deterministic-corpus-comparison" if role == "similar_case" else settings.model_name,
                "status": "complete",
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
) -> dict[str, Any]:
    case_id = case_id or str(uuid4())
    initial_state: CaseState = {
        "case_id": case_id,
        "narrative": narrative.strip(),
        "jurisdiction": jurisdiction,
        "incident_date": incident_date,
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
    async with httpx.AsyncClient(timeout=settings.model_timeout_seconds) as client:
        graph = _build_graph(client, settings)
        final_state = await graph.ainvoke(initial_state)

    outputs = final_state["outputs"]
    # The approved risk matrix and source retrieval are not connected yet.
    outputs["risk"]["numeric_score"] = None
    outputs["risk"]["assessment_status"] = "not_scored_no_approved_matrix"
    trace = [
        *final_state["trace"],
        {"step": "deterministic_risk_scoring", "status": "not_configured"},
        {"step": "human_review", "status": "required"},
    ]
    return {
        "case_id": case_id,
        "status": "PRELIMINARY_REQUIRES_HUMAN_REVIEW",
        "workflow": " -> ".join(spec.key for spec in AGENT_SPECS),
        "model": settings.model_name,
        "agent_outputs": outputs,
        "trace": trace,
        "limitations": [
            "Model-backed agent roles share the configured model; role separation is prompt and workflow separation, not separate model deployments.",
            "Historical retrieved cases are analogies only; their findings do not establish the cause of the current incident.",
            "BGE dense retrieval falls back to BM25 when local embedding weights are not available.",
            "Risk scoring is not configured; the Risk Agent must not return a numeric score.",
            "Human review and approval are required before any result is treated as a finding or action.",
        ],
    }
