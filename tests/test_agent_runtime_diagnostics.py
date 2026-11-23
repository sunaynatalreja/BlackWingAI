from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest

from blackwing.agents.base import AgentOutputValidationError
from blackwing.agents.registry import AGENT_BY_ID
from blackwing.agents.runtime import invoke_agent
from blackwing.agents.report.schema import ReportSectionDraft, normalize_report_section
from blackwing.config import Settings


class InvalidJSONClient:
    calls = 0

    async def post(self, url: str, **kwargs):
        self.calls += 1
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={"message": {"content": "raw diagnostic critic response"}},
            request=request,
        )


def test_invalid_agent_response_logs_validation_details_and_response_preview(caplog):
    settings = Settings(
        host="127.0.0.1", port=8000, llm_provider="ollama",
        ollama_base_url="http://localhost:11434", ollama_model="qwen3:8b",
        ollama_timeout_seconds=2, ollama_temperature=0, ollama_think=False,
        ollama_max_output_tokens=512, api_base_url="", api_model="", api_key="",
        api_timeout_seconds=2, api_temperature=0, api_max_output_tokens=512,
        ui_api_url="http://127.0.0.1:8000", ui_request_timeout_seconds=2,
        rag_embedding_model="BAAI/bge-small-en-v1.5", model_response_log_limit=100,
    )
    client = InvalidJSONClient()
    with caplog.at_level(logging.WARNING, logger="blackwing.agents.runtime"):
        with pytest.raises(AgentOutputValidationError) as error:
            asyncio.run(invoke_agent(
                AGENT_BY_ID["critic"], client=client, settings=settings,
                narrative="Synthetic runway incident narrative.", evidence=[], prior_outputs={},
            ))

    assert isinstance(error.value.__cause__, json.JSONDecodeError)
    assert client.calls == 2  # first attempt plus one schema correction retry
    assert "agent=critic provider=ollama model=qwen3:8b attempt=1/2" in caplog.text
    assert "validation_details" in caplog.text
    assert "raw diagnostic critic response" in caplog.text


def test_report_normalizer_accepts_common_qwen_section_variants():
    normalized = normalize_report_section({
        "executive_summary": "Two accounts differ on vehicle position.",
        "findings": [
            {"claim": "Driver stopped before the access line.", "type": "reported",
             "citations": ["SIM-E1"], "extra": "discarded"},
            "Position remains unverified.",
        ],
        "unused_metadata": "discarded",
    })
    result = ReportSectionDraft.model_validate(normalized).model_dump()
    assert result["summary"] == "Two accounts differ on vehicle position."
    assert result["key_points"][0]["classification"] == "reported_observation"
    assert result["key_points"][0]["evidence_ids"] == ["SIM-E1"]
    assert result["key_points"][1]["classification"] == "verification_gap"
