"""Offline proof of five-agent evidence guards and Markdown report export.

Uses synthetic incident and mock outputs. NO hosted/local model or RAG calls.
Run: python scripts/demo_five_agents.py
"""
from __future__ import annotations

import json
from pathlib import Path

from blackwing.agents.investigation.grounding import ground_investigation_output
from blackwing.agents.similar_case.agent import build_output
from blackwing.agents.mitigation.grounding import ground_mitigation_output
from blackwing.agents.critic.grounding import audit_critic_output
from blackwing.agents.verification.grounding import ground_verification_output
from blackwing.agents.risk.evidence import ground_risk_output
from blackwing.agents.compliance.agent import unassessed_output
from blackwing.reporting import render_markdown_report


def build_synthetic_report() -> dict:
    """A schema-valid demo, not an actual airport investigation."""
    factual = {"document_id": "SYN-FACT", "chunk_id": "SYN-FACT:1", "source_id": "SYN-PRIMARY",
               "incident_id": "SYNTH-123", "source_type": "official_factual",
               "evidence_role": "case_factual", "page_number": 1, "excerpt": "Synthetic runway conflict was recorded."}
    guidance = {"document_id": "SYN-GUIDE", "chunk_id": "SYN-GUIDE:1", "source_id": "SYN-GUIDE",
                "source_type": "advisory", "evidence_role": "general_guidance",
                "publication_date": "2020-01-01", "excerpt": "Synthetic guidance to check runway signs."}
    outputs = {}
    outputs["investigation"] = ground_investigation_output({
        "incident_summary": "Synthetic event: two aircraft entered potentially conflicting runway movements.",
        "documented_facts": [{"text": "A synthetic runway conflict was recorded.",
                              "evidence_state": "documented_fact", "evidence_ids": ["SYN-FACT:1"]}],
        "unknowns": ["Tower and flight-crew timing must be confirmed."],
    }, [factual], incident_id="SYNTH-123")
    outputs["similar_case"] = build_output("Synthetic runway conflict", [], current_incident_id="SYNTH-123")
    outputs["compliance"] = unassessed_output()
    outputs["risk"] = ground_risk_output({
        "hazards": [{"hazard_id": "HAZ-001", "description": "Potential conflicting runway occupancy",
                     "evidence_ids": ["SYN-FACT:1"], "existing_controls": ["Runway clearance procedures"],
                     "control_gaps": ["Control effectiveness unknown"]}],
    }, [factual, guidance])
    outputs["mitigation"] = ground_mitigation_output({
        "recommendations": [{"hazard_ids": ["HAZ-001"],
                             "action": "Review clearance verification and documented surface-movement barriers",
                             "classification": "preventive", "priority": "proposed",
                             "effectiveness_indicator": "Check documented barrier verification rate"}],
        "assumptions": ["Synthetic demonstration only; no real control outcome established."],
    }, outputs)
    outputs["critic"] = audit_critic_output({"issues": []}, outputs)
    outputs["verification"] = ground_verification_output({
        "claims_needing_source": [], "unsupported_conclusions": [], "conflicts": [],
        "completeness_status": "incomplete", "human_approval_required": True,
    }, outputs)
    return {"case_id": "SYNTH-123", "status": "PRELIMINARY_REQUIRES_HUMAN_REVIEW",
            "workflow": "investigation -> similar_case -> compliance -> risk -> mitigation -> critic -> verification",
            "model": "synthetic-mock-no-LLM", "agent_outputs": outputs,
            "trace": [{"step": role, "status": "complete", "model": "mock"} for role in outputs],
            "limitations": ["Synthetic demo is not a factual runway-incursion investigation.",
                            "No model or hosted API was called.", "Qualified investigator review is mandatory."]}


def main():
    outdir = Path("work")
    outdir.mkdir(parents=True, exist_ok=True)
    report = build_synthetic_report()
    json_path = outdir / "five_agents_synthetic_mock_report.json"
    md_path = outdir / "five_agents_synthetic_mock_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    print("Generated synthetic offline report:", json_path, md_path)
    print("Agent roles:", ", ".join(report["agent_outputs"]))
    print("Human approval:", report["agent_outputs"]["verification"]["human_approval_required"])


if __name__ == "__main__":
    main()
