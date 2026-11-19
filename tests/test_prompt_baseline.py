from __future__ import annotations

import csv
import json
import pytest

from blackwing.rag.prompt_baseline import (
    STRATEGIES, SCHEMA_KEYS, build_messages, load_development_tasks,
    run_experiment, summarize, validate_report,
)


def task():
    return {"task_id":"DEV-A", "incident_id":"CASE-X", "incident_date":"2024-01-01",
            "jurisdiction":"IN", "narrative":"Crew reported stopping before holding point.",
            "task_instruction":"Draft a report."}


def report():
    return {"incident_overview":"Unverified narrative says crew stopped before a hold point.",
            "chronology":["Crew stopped."], "potential_hazards":["Potential surface movement conflict"],
            "risk_evidence":"No numerical severity/risk evidence supplied.",
            "compliance_review":"Insufficient evidence to determine applicable requirement.",
            "proposed_mitigations":["Review pilot reports."],
            "missing_evidence":["ATC recording"], "uncertainty":"Narrative only.",
            "source_basis":"user_narrative_only", "human_review_required":True}


def test_three_single_agent_prompts_same_input_no_rag():
    for strategy in STRATEGIES:
        msg = build_messages(strategy, task())
        assert len(msg) == 2
        assert 'Crew reported stopping' in msg[1]['content']
        assert 'CASE-X' in msg[1]['content']
        assert 'no RAG' in msg[0]['content'] or 'No RAG' in msg[0]['content']
        assert 'JFK' not in '\n'.join(x['content'] for x in msg)
    assert 'SYNTHETIC' in build_messages('few_shot', task())[0]['content']
    assert 'internal structured-check' in build_messages('decomposition', task())[0]['content']


def test_only_explicitly_approved_development_rows_and_no_gold_leak(tmp_path):
    p = tmp_path / 'development.csv'
    with p.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[*task().keys(), 'split', 'approved', 'gold_answer', 'private_notes'])
        writer.writeheader()
        writer.writerow({**task(), 'split':'development','approved':'true',
                         'gold_answer':'SECRET GOLD', 'private_notes':'SECRET REVIEW'})
    loaded = load_development_tasks(p)
    assert loaded == [task()]
    assert 'SECRET' not in str(loaded)
    # Fail closed on held out and not approved.
    text = p.read_text().replace('development','final_test')
    p.write_text(text)
    with pytest.raises(ValueError): load_development_tasks(p)
    p.write_text(text.replace('final_test','development').replace('true','false'))
    with pytest.raises(ValueError): load_development_tasks(p)


def test_validator_rejects_invalid_or_misleading_outputs():
    assert validate_report(json.dumps(report()))[0] is True
    assert validate_report('```json\n'+json.dumps(report())+'\n```')[2] == 'malformed_json'
    bad = report(); bad['source_basis'] = 'official_NTSB'
    assert validate_report(json.dumps(bad))[2] == 'source_or_review_flag'
    bad = report(); bad['human_review_required'] = False
    assert not validate_report(json.dumps(bad))[0]
    bad = report(); bad['invented_citation'] = 'FAA 123'
    assert validate_report(json.dumps(bad))[2] == 'schema_keys'


def test_dry_run_does_not_call_model_or_claim_scores():
    def no_call(_):
        raise AssertionError('Offline must never call provider')
    records = run_experiment([task()], provider='ollama', invoke=no_call)
    assert len(records) == 3
    assert all(r['status'] == 'PREVIEW_NOT_EXECUTED' and r['schema_valid'] is None for r in records)
    summary = summarize(records)
    assert summary['strategies']['few_shot']['schema_validity_rate'] is None
    assert summary['human_accuracy'] is None


def test_mocked_live_compares_all_three_and_logs_tokens_without_qa():
    seen = []
    def fake(messages):
        seen.append(messages)
        return json.dumps(report()), {'model':'fake-qwen', 'prompt_tokens':88,'completion_tokens':123}
    rows = run_experiment([task()], provider='ollama', invoke=fake, live=True,
                          strict_token_preflight=False)
    assert len(seen) == 3 and all(len(m) == 2 for m in seen)
    assert {r['strategy'] for r in rows} == set(STRATEGIES)
    assert all(r['status'] == 'COMPLETED' and r['completion_tokens'] == 123 for r in rows)
    assert summarize(rows)['strategies']['zero_shot']['schema_validity_rate'] == 1
    assert summarize(rows)['strategies']['zero_shot']['human_factual_accuracy'] is None


def test_budget_and_provider_guard():
    with pytest.raises(ValueError):
        run_experiment([task()], provider='ollama', max_output=1025)
    with pytest.raises(ValueError):
        run_experiment([task()], provider='bad')
