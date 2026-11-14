"""Load application settings from a properties file selected via environment."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


_ENV_VALUE = re.compile(r"^\$\{([A-Z0-9_]+)(?::-([^}]*))?\}$")


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
    api_key: str
    api_timeout_seconds: float
    api_temperature: float
    api_max_output_tokens: int
    ui_api_url: str
    ui_request_timeout_seconds: float
    rag_embedding_model: str

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
        required = {"app.host", "app.port", "llm.provider", "ui.api_url", "ui.request_timeout_seconds", "rag.embedding_model"}
        provider = values.get("llm.provider", "").lower()
        if not provider:
            raise ValueError(
                "BLACKWING_LLM_PROVIDER is not set in this process. Set it to 'ollama' "
                "or 'openai_compatible' before starting the API and UI."
            )
        if provider == "ollama":
            required.update({"ollama.base_url", "ollama.model", "ollama.timeout_seconds", "ollama.temperature", "ollama.think", "ollama.max_output_tokens"})
        elif provider == "openai_compatible":
            required.update({"api.base_url", "api.model", "api.key", "api.timeout_seconds", "api.temperature", "api.max_output_tokens"})
        else:
            raise ValueError(
                "Invalid BLACKWING_LLM_PROVIDER value. Set it to exactly 'ollama' "
                "or 'openai_compatible' before starting the API and UI."
            )
        missing = required.difference(values)
        missing.update(key for key in required.intersection(values) if not values[key])
        if missing:
            raise ValueError(f"Missing properties: {', '.join(sorted(missing))}")
        return cls(
            host=values["app.host"],
            port=int(values["app.port"]),
            llm_provider=provider,
            ollama_base_url=values.get("ollama.base_url", "").rstrip("/"),
            ollama_model=values.get("ollama.model", ""),
            ollama_timeout_seconds=float(values.get("ollama.timeout_seconds") or 120),
            ollama_temperature=float(values.get("ollama.temperature") or 0.1),
            ollama_think=(values.get("ollama.think") or "false").lower() == "true",
            ollama_max_output_tokens=int(values.get("ollama.max_output_tokens") or 1024),
            api_base_url=values.get("api.base_url", "").rstrip("/"),
            api_model=values.get("api.model", ""),
            api_key=values.get("api.key", ""),
            api_timeout_seconds=float(values.get("api.timeout_seconds") or 120),
            api_temperature=float(values.get("api.temperature") or 0.1),
            api_max_output_tokens=int(values.get("api.max_output_tokens") or 1024),
            ui_api_url=values["ui.api_url"].rstrip("/"),
            ui_request_timeout_seconds=float(values["ui.request_timeout_seconds"]),
            rag_embedding_model=values["rag.embedding_model"],
        )
