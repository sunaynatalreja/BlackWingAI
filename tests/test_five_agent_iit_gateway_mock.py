"""IIT-compatible gateway tests without credentials/network/model downloads."""
import asyncio
import json

import httpx

from blackwing.agents import runtime
from blackwing.agents.verification.agent import SPEC
from blackwing.config import Settings


def settings():
    return Settings(host="127.0.0.1", port=8001, llm_provider="openai_compatible",
        ollama_base_url="", ollama_model="", ollama_timeout_seconds=1,
        ollama_temperature=0, ollama_think=False, ollama_max_output_tokens=512,
        api_base_url="https://iit.example.invalid/v1", api_model="qwen3-8b",
        api_key="fake-token-do-not-use", api_timeout_seconds=3,
        api_temperature=0, api_max_output_tokens=512,
        ui_api_url="http://127.0.0.1:8001", ui_request_timeout_seconds=1,
        rag_embedding_model="BAAI/bge-small-en-v1.5")


def test_hosted_json_gateway_is_configurable_and_auth_header_is_used(monkeypatch):
    monkeypatch.setattr(runtime, "build_small_messages", lambda **kwargs:
                        [{"role": "system", "content": "Return JSON"},
                         {"role": "user", "content": "synthetic"}])
    monkeypatch.setenv("BLACKWING_API_JSON_RESPONSE_FORMAT", "0")
    monkeypatch.setenv("BLACKWING_IIT_MIN_INTERVAL_SECONDS", "1.05")
    sent = []
    def handler(request):
        body = json.loads(request.content)
        sent.append((request, body))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "human_approval_required": True, "completeness_status": "incomplete"})}}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await runtime.invoke_agent(SPEC, client=client, settings=settings(),
                narrative="synthetic case", evidence=[], prior_outputs={})
    result = asyncio.run(run())
    assert result["human_approval_required"] is True
    assert len(sent) == 1
    assert "response_format" not in sent[0][1]
    assert sent[0][0].headers["Authorization"].startswith("Bearer ")
    assert sent[0][1]["model"] == "qwen3-8b"


def test_gateway_transient_failure_retries_with_bound(monkeypatch):
    monkeypatch.setenv("BLACKWING_IIT_MIN_INTERVAL_SECONDS", "1.05")
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(503 if len(calls) == 1 else 200, json={"choices": []})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await runtime._hosted_post(client, url="https://iit.example.invalid/v1/chat/completions",
                                              headers={"Authorization": "Bearer fake"}, payload={"model": "x"})
    answer = asyncio.run(run())
    assert answer.status_code == 200
    assert len(calls) == 2
