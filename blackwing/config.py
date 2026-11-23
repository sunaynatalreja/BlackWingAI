"""Load application settings from a properties file selected via environment."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


_ENV_VALUE = re.compile(r"^\$\{([A-Z0-9_]+)(?::-([^}]*))?\}$")


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
    gold_query_file: str = ""

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
        provider = values.get("llm.provider", "").lower()
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
        embedding_backend = values.get("rag.backend", "iit").lower()
        if embedding_backend == "iit":
            required.update({"rag.embedding_url", "rag.embedding_api_key", "rag.embedding_ca_cert", "rag.embedding_timeout_seconds", "rag.embedding_retries"})
            missing = required.difference(values)
        missing.update(key for key in required.intersection(values) if not values[key])
        if missing:
            raise ValueError(f"Missing properties: {', '.join(sorted(missing))}")
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
                quota = int(value)
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
            port=int(values["app.port"]),
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
            api_timeout_seconds=float(values.get("api.timeout_seconds") or 120),
            api_temperature=float(values.get("api.temperature") or 0.1),
            api_max_output_tokens=int(values.get("api.max_output_tokens") or 1024),
            api_ca_cert=cert_path,
            api_json_response_format=values.get("api.json_response_format", "false").lower() == "true",
            api_min_interval_seconds=float(values.get("api.min_interval_seconds") or 1.1),
            tokenizer_model=values["llm.tokenizer_model"],
            ui_api_url=values["ui.api_url"].rstrip("/"),
            ui_request_timeout_seconds=float(values["ui.request_timeout_seconds"]),
            rag_embedding_model=values["rag.embedding_model"],
            rag_index_dir=values.get("rag.index_dir", "").strip(),
            rag_backend=embedding_backend,
            rag_embedding_url=values.get("rag.embedding_url", "").rstrip("/"),
            rag_embedding_api_key=values.get("rag.embedding_api_key", ""),
            rag_embedding_ca_cert=embedding_cert_path,
            rag_embedding_timeout_seconds=float(values.get("rag.embedding_timeout_seconds") or 60),
            rag_embedding_retries=min(5, max(0, int(values.get("rag.embedding_retries") or 2))),
            rag_cache_path=values.get("rag.cache_path", "").strip(),
            agent_token_quotas=agent_token_quotas,
            gold_query_file=values.get("gold.query_file", "").strip(),
            model_response_log_limit=max(0, int(values.get("logging.model_response_log_limit") or 4000)),
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
