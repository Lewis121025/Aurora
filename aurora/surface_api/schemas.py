from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class InputRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_text: str = Field(
        min_length=1,
        validation_alias=AliasChoices("user_text", "text"),
        serialization_alias="user_text",
    )
    language: str = "auto"


class InputResponse(BaseModel):
    event_id: str
    output_text: str | None
    outcome: str
    next_wake_at: str | None


class HealthResponse(BaseModel):
    version: str
    substrate_alive: bool
    sealed_state_version: str
    anchor_count: int
    next_wake_at: str | None
    last_error: str | None
    provider_healthy: bool


class IntegrityResponse(BaseModel):
    version: str
    runtime_boundary: str
    substrate_transport: str
    sealed_state_version: str
    config_fingerprint: str
    generated_at: str
