from blackwing.config import Settings, _as_float, _as_int


def test_numeric_settings_accept_shell_pasted_curly_quotes(monkeypatch, tmp_path):
    cert = tmp_path / "ca.crt"
    cert.write_text("test certificate placeholder", encoding="utf-8")
    properties = tmp_path / "blackwing.properties"
    properties.write_text("\n".join((
        "app.host=${TEST_HOST}", "app.port=${TEST_PORT}",
        "llm.provider=${TEST_PROVIDER}", "api.base_url=${TEST_BASE}",
        "api.model=${TEST_MODEL}", "api.key=${TEST_KEY}", "api.ca_cert=${TEST_CERT}",
        "api.timeout_seconds=${TEST_TIMEOUT}", "api.temperature=${TEST_TEMPERATURE}",
        "api.max_output_tokens=${BLACKWING_LLM_API_MAX_OUTPUT_TOKENS}",
        "api.json_response_format=${TEST_JSON}", "api.min_interval_seconds=${TEST_INTERVAL}",
        "ui.api_url=${TEST_UI_URL}", "ui.request_timeout_seconds=${TEST_UI_TIMEOUT}",
        "rag.embedding_model=${TEST_EMBEDDING_MODEL}", "rag.backend=${TEST_RAG_BACKEND}",
        "logging.model_response_log_limit=${TEST_LOG_LIMIT}",
        "llm.tokenizer_model=${TEST_TOKENIZER}",
        "agents.token_quota.default=${BLACKWING_AGENT_TOKEN_QUOTA_DEFAULT}",
    )), encoding="utf-8")
    for key, value in {
        "BLACKWING_CONFIG": str(properties), "TEST_HOST": "127.0.0.1", "TEST_PORT": "8000",
        "TEST_PROVIDER": "openai_compatible", "TEST_BASE": "https://example.invalid/v1",
        "TEST_MODEL": "test-model", "TEST_KEY": "test-key", "TEST_CERT": str(cert),
        "TEST_TIMEOUT": "120", "TEST_TEMPERATURE": "0.1", "BLACKWING_LLM_API_MAX_OUTPUT_TOKENS": "“1024”",
        "TEST_JSON": "true", "TEST_INTERVAL": "1.1", "TEST_UI_URL": "http://127.0.0.1:8000",
        "TEST_UI_TIMEOUT": "1800", "TEST_EMBEDDING_MODEL": "bge-small-en-v1.5",
        "TEST_RAG_BACKEND": "none", "TEST_LOG_LIMIT": "4000", "TEST_TOKENIZER": "Qwen/Qwen3-8B",
        "BLACKWING_AGENT_TOKEN_QUOTA_DEFAULT": "“768”",
    }.items():
        monkeypatch.setenv(key, value)
    settings = Settings.load()
    assert settings.api_max_output_tokens == 1024
    assert settings.agent_token_quotas["default"] == 768
    assert _as_int("“512”", name="test") == 512
    assert _as_float("“0.1”", name="test") == 0.1
