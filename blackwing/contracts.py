"""Shared base model for API and agent schemas."""

from pydantic import BaseModel, ConfigDict


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)
