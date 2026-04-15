# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack_api import json_schema_type
from pydantic import BaseModel, Field, HttpUrl, SecretStr

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig


class RunpodProviderDataValidator(BaseModel):
    """Validates provider-specific request data for RunPod inference."""

    runpod_api_token: SecretStr | None = Field(
        default=None,
        description="API token for RunPod models",
    )


@json_schema_type
class RunpodImplConfig(RemoteInferenceProviderConfig):
    """Configuration for the RunPod inference provider."""

    base_url: HttpUrl | None = Field(
        default=None,
        description="The URL for the Runpod model serving endpoint",
    )
    auth_credential: SecretStr | None = Field(
        default=None,
        alias="api_token",
        description="The API token",
    )

    @classmethod
    def sample_run_config(cls, **kwargs: Any) -> dict[str, Any]:
        return {
            "base_url": "${env.RUNPOD_URL:=}",
            "api_token": "${env.RUNPOD_API_TOKEN:=}",
        }
