"""Exercise the Community Cloud entrypoint without live inference or local storage."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import httpx
from openai import OpenAI
from streamlit.testing.v1 import AppTest

import blackwing_app
import copilot_state
import inference

ENTRYPOINT = Path(__file__).parent / "cloud" / "streamlit_app.py"
JFK = "JFK-DCA23LA125-DEMO"


def open_jfk(app):
    next(button for button in app.button if button.key == "home-history").click().run()
    next(item for item in app.selectbox if item.label == "Historical incident").set_value(JFK).run()
    next(button for button in app.button if button.label == "Open historical incident").click().run()
    return app


class CloudTest(TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.addCleanup(patch.stopall)
        root = Path(folder.name)
        patch.object(blackwing_app, "OUTPUT", root).start()
        patch.object(inference, "OUTPUT", root).start()
        patch.object(copilot_state, "DATABASE", root / "workstation.sqlite3").start()
        patch.dict(os.environ, {
            "BLACKWING_API_BASE_URL": "https://iit-gateway.invalid/v1",
            "BLACKWING_API_KEY": "cloud-test-placeholder",
            "BLACKWING_MODEL": "qwen3-8b",
            "BLACKWING_ALLOW_REMOTE": "YES",
            "BLACKWING_TEAM_KEY_COORDINATED": "YES",
            "BLACKWING_IIT_RESPONSE_FORMAT": "json_schema",
        }).start()

    def test_browse_with_no_credentials_and_no_local_provider(self):
        with patch.dict(os.environ, {"BLACKWING_API_KEY": "", "BLACKWING_ALLOW_REMOTE": "NO"}):
            with patch("openai.OpenAI") as connection:
                app = open_jfk(AppTest.from_file(str(ENTRYPOINT), default_timeout=20).run())
                connection.assert_not_called()
        self.assertFalse(app.exception)
        self.assertIn("Hazards and risk", [tab.label for tab in app.tabs])
        self.assertTrue(any("Not configured" in item.value for item in app.text))
        self.assertTrue(any("Retrieval · BM25" == item.value for item in app.caption))
        self.assertFalse(any(item.label == "Inference provider" for item in app.selectbox))
        self.assertEqual(len(blackwing_app.load_index()), 424)

    def test_hosted_hazard_review_pdf_and_session_isolation(self):
        calls = []

        def respond(request):
            payload = json.loads(request.content)
            calls.append(payload)
            data = json.loads(payload["messages"][1]["content"])
            evidence_id, quote_id, source = next(
                (evidence_id, quote_id, quote)
                for evidence_id, scope, options in data["passages"] if scope == "direct"
                for quote_id, quote in options.items()
                if "annunciated a runway occupied alert" in quote
            )
            result = {"answer": "Potential runway conflict.", "findings": [{
                "claim": "ASDE-X annunciated a runway occupied alert.",
                "uncertainty": "The alert identifies a potential runway conflict.",
                "specialty": "hazards",
                "potential_consequence": "Aircraft collision if the conflict persists.",
                "existing_controls": ["annunciated a runway occupied alert"],
                "citations": [{"id": evidence_id, "quote": quote_id}],
            }]}
            self.assertIn("annunciated a runway occupied alert", source)
            return httpx.Response(200, json={
                "id": "fixture", "object": "chat.completion", "created": 1, "model": "qwen3-8b",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": json.dumps(result)}}],
                "usage": {"prompt_tokens": 220, "completion_tokens": 80, "total_tokens": 300},
            })

        def connect(**kwargs):
            kwargs["http_client"].close()
            kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(respond))
            return OpenAI(**kwargs)

        with patch("openai.OpenAI", side_effect=connect), patch(
            "copilot.local_call", side_effect=AssertionError("Cloud must use IIT")
        ):
            app = open_jfk(AppTest.from_file(str(ENTRYPOINT), default_timeout=20).run())
            self.assertFalse(app.exception)
            question = "What hazards are indicated by the annunciated runway occupied alert?"
            next(item for item in app.text_area if item.label == "Ask about this incident").set_value(question)
            next(button for button in app.button if button.label == "Send question").click().run()
            self.assertTrue(any("Authorize the IIT transfer" in item.value for item in app.error))
            self.assertEqual(len(calls), 0)

            next(item for item in app.text_area if item.label == "Ask about this incident").set_value(question)
            app.checkbox[0].check()
            next(button for button in app.button if button.label == "Send question").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["response_format"]["type"], "json_schema")
            self.assertTrue(any("Aircraft collision if the conflict persists" in item.value for item in app.text))
            self.assertTrue(any("Tokens used · 300" == item.value for item in app.text))
            next(item for item in app.text_input if item.label == "Reviewer name").set_value("Cloud test reviewer")
            next(item for item in app.checkbox if item.label.startswith("I checked")).check()
            next(button for button in app.button if button.label == "Accept finding").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any(item.label == "Download preliminary report" for item in app.get("download_button")))

            other = open_jfk(AppTest.from_file(str(ENTRYPOINT), default_timeout=20).run())
            self.assertFalse(other.exception)
            self.assertFalse(any(question in item.value for item in other.text))
            self.assertFalse(any(item.label == "Download preliminary report" for item in other.get("download_button")))
        self.assertEqual(len(calls), 1)
        self.assertFalse(copilot_state.DATABASE.exists())
