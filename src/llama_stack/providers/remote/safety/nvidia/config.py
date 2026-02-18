# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from llama_stack_api import json_schema_type

_LEGACY_API_MODE_MAP = {
    "microservice": "guardrail_checks",
    "openai": "guardrail_chat_completions",
}


class GuardrailsApiMode(str, Enum):
    GUARDRAIL_CHECKS = "guardrail_checks"
    GUARDRAIL_CHAT_COMPLETIONS = "guardrail_chat_completions"


@json_schema_type
class NVIDIASafetyConfig(BaseModel):
    guardrails_service_url: str = Field(
        default_factory=lambda: os.getenv("GUARDRAILS_SERVICE_URL", "http://0.0.0.0:7331"),
        description="The URL for accessing the NeMo Guardrails service",
    )
    config_id: str = Field(
        default_factory=lambda: os.getenv("NVIDIA_GUARDRAILS_CONFIG_ID", "self-check"),
        description="Guardrails configuration ID to use from the configuration store",
        min_length=1,
    )
    blocked_message: str = Field(
        default_factory=lambda: os.getenv("NVIDIA_GUARDRAILS_BLOCKED_MESSAGE", "I'm sorry, I can't respond to that."),
        description="The message NeMo Guardrails returns when input is blocked",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the guardrails model, between 0 and 2",
    )
    timeout: int = Field(
        default=60,
        description="Timeout in seconds for HTTP requests to the guardrails service",
    )
    api_mode: GuardrailsApiMode = Field(
        default=GuardrailsApiMode.GUARDRAIL_CHECKS,
        description="API mode: 'guardrail_checks' for /v1/guardrail/checks, 'guardrail_chat_completions' for /v1/guardrail/chat/completions",
    )

    @field_validator("api_mode", mode="before")
    @classmethod
    def _normalize_api_mode(cls, v: str) -> str:
        return _LEGACY_API_MODE_MAP.get(v, v) if isinstance(v, str) else v

    @classmethod
    def sample_run_config(cls, **kwargs) -> dict[str, Any]:
        return {
            "guardrails_service_url": "${env.GUARDRAILS_SERVICE_URL:=http://localhost:7331}",
            "config_id": "${env.NVIDIA_GUARDRAILS_CONFIG_ID:=self-check}",
            "blocked_message": "${env.NVIDIA_GUARDRAILS_BLOCKED_MESSAGE:=I'm sorry, I can't respond to that.}",
            "temperature": 1.0,
            "timeout": 60,
            "api_mode": "${env.NVIDIA_GUARDRAILS_API_MODE:=guardrail_checks}",
        }
