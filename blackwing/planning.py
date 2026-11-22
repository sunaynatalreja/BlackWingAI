"""Deterministic investigation planning and evidence-gated correction policy."""
from blackwing.agents.critic.grounding import audit_critic_output
from blackwing.case_inputs import synthetic_risk_unassessed

MAX_EXECUTION_STEPS = 10
CORRECTABLE = {"ungrounded_fact": "investigation", "ungrounded_hazard": "risk",
               "unlinked_mitigation": "mitigation", "no_grounded_mitigations": "mitigation"}


def plan_investigation(state):
    packets = state.get("agent_evidence")
    evidence = state.get("evidence", [])
    def sources(role):
        return packets.get(role, []) if packets is not None else evidence
    has_rules = any(str(x.get(key) or "").casefold() in {"procedure", "regulation", "operational_order"}
                    for x in sources("compliance") for key in ("source_type", "document_type"))
    modes = {"investigation": "model_with_grounding", "similar_case": "deterministic_tool",
             "compliance": "model_with_grounding" if has_rules else "unassessed",
             "risk": "model_then_reference_tool_per_hazard", "mitigation": "model_with_hazard_links",
             "critic": "model_and_deterministic_audit", "verification": "model_and_deterministic_checks",
             "report": "deterministic_assembly"}
    abstain_risk = synthetic_risk_unassessed(state, sources("risk"))
    if abstain_risk:
        modes["risk"] = "unassessed_synthetic"
    return {"task_modes": modes, "tool_requirements": {
                "retrieve_evidence": "preflight retrieval when requested; use supplied evidence otherwise",
                "find_similar_incidents": "compare supplied historical records, including explicit no-match results",
                "assess_reference_risk": "execute only for returned hazards after citation validation"}, "tasks": ["investigation", "similar_case", "compliance", "risk", "mitigation", "critic", "verification", "report"],
            "tools": ["retrieve_evidence", "find_similar_incidents", "assess_reference_risk"],
            "decisions": {"investigation": "ground supplied evidence; preserve unknowns",
                          "similar_case": "compare historical records separately",
                          "compliance": "compare authoritative rules" if has_rules else "unassessed: authoritative rules missing",
                          "risk": "unassessed: unverified synthetic accounts provide no rated-input basis" if abstain_risk else "score only supported severity and likelihood",
                          "correction": "one evidence-gated target; unresolved issues require human review"},
            "source_counts": {role: len(sources(role)) for role in ("investigation", "similar_case", "compliance", "risk")},
            "max_execution_steps": MAX_EXECUTION_STEPS, "max_corrections": 1}


def correction_target(state):
    if state.get("correction_attempts", 0) or state.get("execution_steps", 0) >= MAX_EXECUTION_STEPS - 2:
        return None
    outputs = state.get("outputs", {})
    for issue in outputs.get("critic", {}).get("issues", []):
        role = CORRECTABLE.get(issue.get("issue_type"))
        if not role or issue.get("affected_output") != role:
            continue
        if role == "mitigation":
            if any(h.get("evidence_ids") for h in outputs.get("risk", {}).get("hazards", [])):
                return role
        else:
            packets = state.get("agent_evidence")
            evidence = packets.get(role, []) if packets is not None else state.get("evidence", [])
            if any(str(e.get("source_type") or e.get("document_type")) in {"official_factual", "official_preliminary", "user_submitted", "user_report"}
                   and e.get("evidence_role") not in {"historical_analogy", "historical"}
                   and not str(e.get("chunk_id") or e.get("evidence_id") or "").startswith("CASE-NARRATIVE:") for e in evidence):
                return role
    return None


def review_correction(before, outputs):
    """Recheck only issues whose predicates are deterministic; keep model concerns."""
    deterministic = audit_critic_output({"issues": []}, outputs)
    current = {(i["issue_type"], i.get("affected_output")) for i in deterministic["issues"]}
    retained = []
    resolved = []
    for issue in before.get("issues", []):
        key = (issue["issue_type"], issue.get("affected_output"))
        if issue["issue_type"] in CORRECTABLE and key not in current:
            resolved.append(issue)
        else:
            retained.append(issue)
    return audit_critic_output({**before, "issues": retained}, outputs), resolved
