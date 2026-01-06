# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from llama_stack_api import json_schema_type


class GuardrailsApiMode(str, Enum):
    """API mode for NeMo Guardrails service."""

    MICROSERVICE = "microservice"  # Enterprise NIM: /v1/guardrail/checks
    OPENAI = "openai"  # Open-source toolkit: /v1/guardrail/chat/completions


@json_schema_type
class NVIDIASafetyConfig(BaseModel):
    """
    Configuration for the NVIDIA Guardrail microservice endpoint.

    Attributes:
        guardrails_service_url (str): A base url for accessing the NVIDIA guardrail endpoint, e.g. http://0.0.0.0:7331
        config_id (str): The ID of the guardrails configuration to use from the configuration store
         (https://developer.nvidia.com/docs/nemo-microservices/guardrails/source/guides/configuration-store-guide.html)

    """

    guardrails_service_url: str = Field(
        default_factory=lambda: os.getenv("GUARDRAILS_SERVICE_URL", "http://0.0.0.0:7331"),
        description="The URL for accessing the NeMo Guardrails service",
    )
    config_id: str | None = Field(
        default_factory=lambda: os.getenv("NVIDIA_GUARDRAILS_CONFIG_ID", "self-check"),
        description="Guardrails configuration ID to use from the configuration store",
    )
    blocked_message: str = Field(
        default_factory=lambda: os.getenv("NVIDIA_GUARDRAILS_BLOCKED_MESSAGE", "I'm sorry, I can't respond to that."),
        description="The message NeMo Guardrails returns when input is blocked",
    )
    temperature: float = Field(
        default=1.0,
        description="Sampling temperature for the guardrails model, between 0 and 2",
    )
    api_mode: GuardrailsApiMode = Field(
        default=GuardrailsApiMode.MICROSERVICE,
        description="API mode: 'microservice' for enterprise NIM (/v1/guardrail/checks), 'openai' for open-source toolkit (/v1/guardrail/chat/completions)",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> dict[str, Any]:
        return {
            "guardrails_service_url": "${env.GUARDRAILS_SERVICE_URL:=http://localhost:7331}",
            "config_id": "${env.NVIDIA_GUARDRAILS_CONFIG_ID:=self-check}",
            "blocked_message": "${env.NVIDIA_GUARDRAILS_BLOCKED_MESSAGE:=I'm sorry, I can't respond to that.}",
            "temperature": 1.0,
            "api_mode": "${env.NVIDIA_GUARDRAILS_API_MODE:=microservice}",
        }
