"""Shared contracts for independently maintained agent packages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Type

from pydantic import BaseModel


@dataclass(frozen=True)
class AgentSpec:
    key: str
    name: str
    prompt_path: Path
    output_schema: Type[BaseModel]
    upstream_context: tuple[str, ...]


class AgentExecutionError(Exception):
    """Public-safe error plus optional bounded details for local server logs."""

    def __init__(self, agent: str, code: str, public_message: str,
                 debug_detail: str | None = None) -> None:
        super().__init__(public_message)
        self.agent = agent
        self.code = code
        self.public_message = public_message
        # Local API logs only; never returned from the public run/status API.
        self.debug_detail = debug_detail


class AgentOutputValidationError(ValueError):
    """Final model-output failure with bounded diagnostics for local server logs."""

    def __init__(self, diagnostic: str, response_preview: str) -> None:
        super().__init__("Model output failed agent schema validation after retry")
        self.diagnostic = diagnostic
        self.response_preview = response_preview
