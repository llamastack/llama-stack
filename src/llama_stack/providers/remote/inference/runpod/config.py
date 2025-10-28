# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, SecretStr

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from llama_stack.schema_utils import json_schema_type


class RunpodProviderDataValidator(BaseModel):
    runpod_api_token: str | None = Field(
        default=None,
        description="API token for RunPod models",
    )


@json_schema_type
class RunpodImplConfig(RemoteInferenceProviderConfig):
    url: str | None = Field(
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
            "url": "${env.RUNPOD_URL:=}",
            "api_token": "${env.RUNPOD_API_TOKEN}",
        }
