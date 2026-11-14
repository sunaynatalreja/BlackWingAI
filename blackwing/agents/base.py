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
    """Safe diagnostic intended for API clients; carries no prompt or response text."""

    def __init__(self, agent: str, code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.agent = agent
        self.code = code
        self.public_message = public_message
