# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, SecretStr

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from llama_stack_api import json_schema_type


class GeminiProviderDataValidator(BaseModel):
    """Validates provider-specific request data for Google Gemini inference."""

    gemini_api_key: SecretStr | None = Field(
        default=None,
        description="API key for Gemini models",
    )


@json_schema_type
class GeminiConfig(RemoteInferenceProviderConfig):
    """Configuration for the Google Gemini inference provider."""

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.GEMINI_API_KEY:=}", **kwargs) -> dict[str, Any]:
        return {
            "api_key": api_key,
        }
