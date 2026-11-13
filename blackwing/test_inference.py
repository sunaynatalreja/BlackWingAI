"""Exercise both inference paths without sending network requests."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import monotonic, sleep
from unittest import TestCase
from unittest.mock import patch

import httpx
from openai import OpenAI

import blackwing_app
import inference

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}},
          "required": ["answer"], "additionalProperties": False}
RESPONSE = {
    "id": "mock-completion", "object": "chat.completion", "created": 1, "model": "qwen3-8b",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"answer":"ok"}'}}],
    "usage": {"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48},
}
HOSTED_ENV = {
    "BLACKWING_API_BASE_URL": "https://iit-gateway.invalid/v1", "BLACKWING_MODEL": "qwen3-8b",
    "BLACKWING_API_KEY": "fixture-key-never-log", "BLACKWING_ALLOW_REMOTE": "YES",
    "BLACKWING_TEAM_KEY_COORDINATED": "YES", "BLACKWING_IIT_QUOTA_HEADER": "X-Team-Tokens",
    "BLACKWING_IIT_RESPONSE_FORMAT": "json_schema", "BLACKWING_USE_RESPONSE_CACHE": "YES",
}


class InferenceTest(TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.addCleanup(patch.stopall)
        patch.object(inference, "OUTPUT", Path(folder.name)).start()
        patch.object(blackwing_app, "OUTPUT", Path(folder.name)).start()
        patch.dict(os.environ, HOSTED_ENV).start()

    def mock_gateway(self, handler):
        def connect(**kwargs):
            kwargs["http_client"].close()
            kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(handler))
            return OpenAI(**kwargs)
        return patch("openai.OpenAI", side_effect=connect)

    def test_hosted_authorization_schema_cache_usage_and_status(self):
        seen = []

        def response(request):
            seen.append((request.url.path, json.loads(request.content)))
            return httpx.Response(200, json=RESPONSE, headers={"X-Team-Tokens": "500"})

        trace = []
        with self.mock_gateway(response) as client:
            with self.assertRaises(inference.InferenceError):
                inference.model_call("Test system", "Test input", 512, SCHEMA, provider="iit", trace=trace)
            self.assertFalse(client.called)
            self.assertEqual(inference.inference_status("iit")["api_status"], "Not checked")
            for _ in range(2):
                self.assertEqual(json.loads(inference.model_call("Test system", "Test input", 512, SCHEMA,
                    provider="iit", authorized=True, trace=trace)), {"answer": "ok"})
        self.assertEqual(len(seen), 1)
        endpoint, body = seen[0]
        self.assertEqual(endpoint, "/v1/chat/completions")
        self.assertEqual(body["response_format"]["json_schema"]["schema"], SCHEMA)
        self.assertEqual(body["max_tokens"], 512)
        self.assertEqual([row["request_sent"] for row in trace], [False, True, False])
        self.assertNotIn(HOSTED_ENV["BLACKWING_API_KEY"], json.dumps(trace))
        status = inference.inference_status("iit")
        self.assertEqual(status["requests_completed"], 1)
        self.assertEqual(status["tokens_used"], 48)
        self.assertEqual(status["remaining_quota"], 500)

    def test_hosted_serializes_requests_and_records_provider_error(self):
        started, active, peak = [], 0, 0
        mutex = Lock()

        def respond(request):
            nonlocal active, peak
            with mutex:
                active += 1
                peak = max(peak, active)
                started.append(monotonic())
            sleep(0.08)
            with mutex:
                active -= 1
            return httpx.Response(429, json={"error": {"message": HOSTED_ENV["BLACKWING_API_KEY"]}})

        def request(number):
            trace = []
            with self.assertRaises(inference.InferenceError) as failure:
                inference.model_call("System", f"Question {number}", 512, SCHEMA,
                                     provider="iit", authorized=True, cache=False, trace=trace)
            self.assertTrue(failure.exception.retryable)
            self.assertEqual(len(trace), 1)
            self.assertEqual(trace[0]["http_status"], 429)
            self.assertNotIn(HOSTED_ENV["BLACKWING_API_KEY"], json.dumps(trace))

        with self.mock_gateway(respond), ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(request, (1, 2)))
        self.assertEqual(peak, 1)
        self.assertGreaterEqual(started[1] - started[0], 1.0)
        status = inference.inference_status("iit")
        self.assertEqual(status["requests_attempted"], 2)
        self.assertEqual(status["requests_completed"], 0)
        self.assertEqual(status["api_status"], "Request rejected")
        self.assertNotIn("remaining_quota", status)

    def test_local_reports_unknown_usage_and_preflight_blocks_excess(self):
        with patch("review_workflow.local_call", return_value=('{"answer":"ok"}', {"status":"ok"})) as local:
            trace = []
            inference.model_call("System", "Question", 512, SCHEMA, provider="local", trace=trace)
            self.assertIsNone(trace[0]["total_tokens"])
            with self.assertRaises(inference.InferenceError):
                inference.model_call("System", "Question", 1025, SCHEMA, provider="local", trace=trace)
            local.assert_called_once()
            self.assertFalse(trace[-1]["request_sent"])
