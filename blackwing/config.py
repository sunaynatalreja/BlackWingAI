"""Load application settings from a properties file selected via environment."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


_ENV_VALUE = re.compile(r"^\$\{([A-Z0-9_]+)(?::-([^}]*))?\}$")
_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’"}


def _unquote_scalar(value: str) -> str:
    """Accept shell values pasted with matching straight or smart quotes.

    This only applies to non-secret scalar settings; API keys remain subject to
    strict validation and are never silently rewritten.
    """
    value = str(value).strip()
    if len(value) >= 2 and _QUOTE_PAIRS.get(value[0]) == value[-1]:
        return value[1:-1].strip()
    return value


def _as_int(value: str, *, name: str) -> int:
    try:
        return int(_unquote_scalar(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; check for extra quote characters in its environment value") from exc


def _as_float(value: str, *, name: str) -> float:
    try:
        return float(_unquote_scalar(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric; check for extra quote characters in its environment value") from exc


def _validate_bearer_key(value: str, *, property_name: str, env_name: str) -> None:
    """Reject formatting artifacts before a key is used in an HTTP header."""
    if not value:
        return
    if not value.isascii():
        raise ValueError(f"{property_name} contains non-ASCII characters; re-enter {env_name} without smart quotes")
    if value.startswith("Bearer "):
        raise ValueError(f"{property_name} must contain only the API key; the client adds the Bearer prefix")
    if value[:1] in {'"', "'"} or value[-1:] in {'"', "'"}:
        raise ValueError(f"{property_name} must not include surrounding quote characters")
    if any(character.isspace() for character in value):
        raise ValueError(f"{property_name} contains whitespace; re-enter {env_name} without spaces")


def _read_properties(path: Path) -> dict[str, str]:
    properties: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith(("#", "!")):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid property at {path}:{line_number}")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        match = _ENV_VALUE.fullmatch(value)
        if not match:
            raise ValueError(f"Property {key!r} must reference an environment variable")
        env_name = match.group(1)
        default_value = match.group(2)
        resolved = os.environ.get(env_name)
        # Provider-specific properties can be unused in the selected mode.
        properties[key] = (resolved if resolved is not None else default_value or "").strip()
    return properties


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    llm_provider: str
    ollama_base_url: str
    ollama_model: str
    ollama_timeout_seconds: float
    ollama_temperature: float
    ollama_think: bool
    ollama_max_output_tokens: int
    api_base_url: str
    api_model: str
    api_key: str = field(repr=False)
    api_timeout_seconds: float
    api_temperature: float
    api_max_output_tokens: int
    ui_api_url: str
    ui_request_timeout_seconds: float
    rag_embedding_model: str
    rag_index_dir: str = ""
    model_response_log_limit: int = 4000
    api_ca_cert: str = ""
    api_json_response_format: bool = False
    api_min_interval_seconds: float = 1.1
    tokenizer_model: str = "Qwen/Qwen3-8B"
    rag_backend: str = "iit"
    rag_embedding_url: str = ""
    rag_embedding_api_key: str = ""
    rag_embedding_ca_cert: str = ""
    rag_embedding_timeout_seconds: float = 60
    rag_embedding_retries: int = 2
    rag_cache_path: str = ""
    agent_token_quotas: dict[str, int] | None = None
    final_report_gold_dir: str = ""
    final_report_bertscore_model: str = "distilbert-base-uncased"
    observability_backend: str = "none"
    langsmith_api_key: str = field(default="", repr=False)
    langsmith_project: str = "BlackWingAI"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    prompt_cost_per_million: float | None = None
    completion_cost_per_million: float | None = None
    knowledge_cache_ttl_seconds: int = 60
    checkpoint_database_path: str = "work/blackwing_checkpoints.sqlite3"

    @property
    def model_name(self) -> str:
        return self.ollama_model if self.llm_provider == "ollama" else self.api_model

    @property
    def model_timeout_seconds(self) -> float:
        return self.ollama_timeout_seconds if self.llm_provider == "ollama" else self.api_timeout_seconds

    @classmethod
    def load(cls) -> "Settings":
        config_path = os.environ.get("BLACKWING_CONFIG")
        if not config_path:
            raise ValueError("Set BLACKWING_CONFIG to the properties file path")
        values = _read_properties(Path(config_path).expanduser())
        required = {"app.host", "app.port", "llm.provider", "ui.api_url", "ui.request_timeout_seconds", "rag.embedding_model", "rag.backend", "logging.model_response_log_limit", "llm.tokenizer_model"}
        provider = _unquote_scalar(values.get("llm.provider", "")).lower()
        if not provider:
            raise ValueError(
                "BLACKWING_LLM_PROVIDER is not set in this process. Set it to 'openai_compatible'."
            )
        if provider == "openai_compatible":
            required.update({"api.base_url", "api.model", "api.key", "api.ca_cert", "api.timeout_seconds", "api.temperature", "api.max_output_tokens", "api.json_response_format", "api.min_interval_seconds"})
        else:
            raise ValueError(
                "Invalid BLACKWING_LLM_PROVIDER value. Set it to exactly 'openai_compatible'."
            )
        missing = required.difference(values)
        embedding_backend = _unquote_scalar(values.get("rag.backend", "iit")).lower()
        if embedding_backend == "iit":
            required.update({"rag.embedding_url", "rag.embedding_api_key", "rag.embedding_ca_cert", "rag.embedding_timeout_seconds", "rag.embedding_retries"})
            missing = required.difference(values)
        missing.update(key for key in required.intersection(values) if not values[key])
        if missing:
            raise ValueError(f"Missing properties: {', '.join(sorted(missing))}")
        if _unquote_scalar(values.get("observability.backend", "none")).lower() not in {"none", "langsmith"}:
            raise ValueError("observability.backend must be 'none' or 'langsmith'")
        if _unquote_scalar(values.get("observability.backend", "none")).lower() == "langsmith" and not values.get("observability.langsmith_api_key", "").strip():
            raise ValueError("Set LANGSMITH_API_KEY when observability.backend=langsmith")
        _validate_bearer_key(values.get("api.key", ""),
                             property_name="api.key", env_name="BLACKWING_LLM_API_KEY")
        _validate_bearer_key(values.get("rag.embedding_api_key", ""),
                             property_name="rag.embedding_api_key", env_name="BLACKWING_EMBEDDING_API_KEY")
        agent_token_quotas: dict[str, int] = {}
        for key, value in values.items():
            prefix = "agents.token_quota."
            if not key.startswith(prefix):
                continue
            agent_name = key[len(prefix):]
            try:
                quota = _as_int(value, name=key)
            except ValueError as exc:
                raise ValueError(f"{key} must be a positive integer") from exc
            if quota < 1:
                raise ValueError(f"{key} must be a positive integer")
            agent_token_quotas[agent_name] = quota
        if "default" not in agent_token_quotas:
            raise ValueError("Missing properties: agents.token_quota.default")
        cert_path = values.get("api.ca_cert", "").strip()
        if provider == "openai_compatible" and not Path(cert_path).expanduser().is_file():
            raise ValueError("api.ca_cert must reference an existing CA certificate file")
        embedding_cert_path = values.get("rag.embedding_ca_cert", "").strip()
        if embedding_backend == "iit" and not Path(embedding_cert_path).expanduser().is_file():
            raise ValueError("rag.embedding_ca_cert must reference an existing CA certificate file")
        if embedding_backend not in {"iit", "local", "none"}:
            raise ValueError("rag.backend must be 'iit', 'local', or 'none'")
        return cls(
            host=values["app.host"],
            port=_as_int(values["app.port"], name="app.port"),
            llm_provider=provider,
            ollama_base_url="",
            ollama_model="",
            ollama_timeout_seconds=120,
            ollama_temperature=0.1,
            ollama_think=False,
            ollama_max_output_tokens=0,
            api_base_url=values.get("api.base_url", "").rstrip("/"),
            api_model=values.get("api.model", ""),
            api_key=values.get("api.key", ""),
            api_timeout_seconds=_as_float(values.get("api.timeout_seconds") or "120", name="api.timeout_seconds"),
            # Temperature zero is a meaningful deterministic setting. Avoid
            # `or 0.1`, which silently changed an explicit 0 back to 0.1.
            api_temperature=_as_float(values["api.temperature"], name="api.temperature"),
            api_max_output_tokens=_as_int(values.get("api.max_output_tokens") or "1024", name="api.max_output_tokens"),
            api_ca_cert=cert_path,
            api_json_response_format=_unquote_scalar(values.get("api.json_response_format", "false")).lower() == "true",
            api_min_interval_seconds=_as_float(values.get("api.min_interval_seconds") or "1.1", name="api.min_interval_seconds"),
            tokenizer_model=values["llm.tokenizer_model"],
            ui_api_url=values["ui.api_url"].rstrip("/"),
            ui_request_timeout_seconds=_as_float(values["ui.request_timeout_seconds"], name="ui.request_timeout_seconds"),
            rag_embedding_model=values["rag.embedding_model"],
            rag_index_dir=values.get("rag.index_dir", "").strip(),
            rag_backend=embedding_backend,
            rag_embedding_url=values.get("rag.embedding_url", "").rstrip("/"),
            rag_embedding_api_key=values.get("rag.embedding_api_key", ""),
            rag_embedding_ca_cert=embedding_cert_path,
            rag_embedding_timeout_seconds=_as_float(values.get("rag.embedding_timeout_seconds") or "60", name="rag.embedding_timeout_seconds"),
            rag_embedding_retries=min(5, max(0, _as_int(values.get("rag.embedding_retries") or "2", name="rag.embedding_retries"))),
            rag_cache_path=values.get("rag.cache_path", "").strip(),
            agent_token_quotas=agent_token_quotas,
            model_response_log_limit=max(0, _as_int(values.get("logging.model_response_log_limit") or "4000", name="logging.model_response_log_limit")),
            final_report_gold_dir=values.get("evaluation.final_report_gold_dir", "").strip(),
            final_report_bertscore_model=values.get("evaluation.bertscore_model", "distilbert-base-uncased").strip(),
            observability_backend=_unquote_scalar(values.get("observability.backend", "none")).lower(),
            langsmith_api_key=values.get("observability.langsmith_api_key", "").strip(),
            langsmith_project=values.get("observability.langsmith_project", "BlackWingAI").strip(),
            langsmith_endpoint=values.get("observability.langsmith_endpoint", "https://api.smith.langchain.com").strip(),
            prompt_cost_per_million=_as_float(values["metrics.prompt_cost_per_million"], name="metrics.prompt_cost_per_million") if values.get("metrics.prompt_cost_per_million") else None,
            completion_cost_per_million=_as_float(values["metrics.completion_cost_per_million"], name="metrics.completion_cost_per_million") if values.get("metrics.completion_cost_per_million") else None,
            knowledge_cache_ttl_seconds=max(0, _as_int(values.get("cache.knowledge_ttl_seconds", "60") or "0", name="cache.knowledge_ttl_seconds")),
            checkpoint_database_path=values.get("workflow.checkpoint_database_path", "work/blackwing_checkpoints.sqlite3").strip(),
        )

    def output_token_quota(self, agent_key: str) -> int:
        """Return an agent-specific output cap, bounded by the provider-wide cap."""
        quotas = self.agent_token_quotas or {}
        quota = quotas.get(agent_key)
        if quota is None and agent_key.startswith("report_"):
            quota = quotas.get("report_section")
        if quota is None:
            quota = quotas.get("default", self.api_max_output_tokens)
        return max(1, min(int(quota), self.api_max_output_tokens))
