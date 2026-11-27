from __future__ import annotations

import asyncio

from blackwing import final_report_evaluation_routes as routes
from blackwing.evaluation import final_report


class _Response:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"finish_reason": "stop", "message": {"content": self._content}}]}


class _Settings:
    api_model = "test-model"
    api_base_url = "https://example.invalid/v1"
    api_key = "test-key"
    api_json_response_format = False
    api_min_interval_seconds = 0

    def output_token_quota(self, _agent):
        return 768


def test_section_judge_retries_malformed_json(monkeypatch):
    replies = iter([
        _Response('{"score": 75, "coverage": 70, "consistency": 80} extra'),
        _Response('{"score":75,"coverage":70,"consistency":80,'
                  '"material_omissions":[],"unsupported_or_conflicting_claims":[],"rationale":"Supported."}'),
    ])
    calls = []

    async def fake_post(_client, **kwargs):
        calls.append(kwargs["payload"])
        return next(replies)

    monkeypatch.setattr(final_report, "_hosted_post", fake_post)
    result = asyncio.run(final_report._judge_section(
        "overview", "Incident overview", "generated", "reference", "gold section", _Settings(), object()))

    assert result["score"] == 75
    assert len(calls) == 2
    assert calls[1]["max_tokens"] >= calls[0]["max_tokens"]
    assert "prior response was malformed" in calls[1]["messages"][0]["content"]


class _Repository:
    def __init__(self, run):
        self.run = run
        self.patches = []

    def patch_run(self, run_id, update):
        self.patches.append((run_id, update))
        self.run.update(update)


def test_completed_evaluation_with_missing_judge_scores_becomes_retryable():
    run = {
        "run_id": "run-1",
        "final_report_evaluation": {
            "status": "completed",
            "sections": {
                "overview": {"status": "completed", "llm_judge": {"score": 80}},
                "findings": {"status": "partial", "llm_judge": None,
                             "cosine": {"cosine_similarity": 0.8}},
            },
        },
    }
    repo = _Repository(run)

    recovered = routes._recover(repo, run)

    assert recovered["status"] == "incomplete"
    assert recovered["sections"]["overview"]["llm_judge"]["score"] == 80
    assert recovered["sections"]["findings"]["cosine"]["cosine_similarity"] == 0.8
    assert repo.patches
