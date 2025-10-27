# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Any

from pydantic import BaseModel, Field

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from llama_stack.schema_utils import json_schema_type


class WatsonXProviderDataValidator(BaseModel):
    watsonx_project_id: str | None = Field(
        default=None,
        description="IBM WatsonX project ID",
    )
    watsonx_api_key: str | None = None


@json_schema_type
class WatsonXConfig(RemoteInferenceProviderConfig):
    url: str = Field(
        default_factory=lambda: os.getenv("WATSONX_BASE_URL", "https://us-south.ml.cloud.ibm.com"),
        description="A base url for accessing the watsonx.ai",
    )
    project_id: str | None = Field(
        default=None,
        description="The watsonx.ai project ID",
    )
    timeout: int = Field(
        default=60,
        description="Timeout for the HTTP requests",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> dict[str, Any]:
        return {
            "url": "${env.WATSONX_BASE_URL:=https://us-south.ml.cloud.ibm.com}",
            "api_key": "${env.WATSONX_API_KEY:=}",
            "project_id": "${env.WATSONX_PROJECT_ID:=}",
        }
