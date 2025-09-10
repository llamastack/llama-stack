# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Any

from pydantic import BaseModel, Field, SecretStr

from llama_stack.schema_utils import json_schema_type


class WatsonXProviderDataValidator(BaseModel):
    url: str
    api_key: str
    project_id: str


@json_schema_type
class WatsonXConfig(BaseModel):
    url: str = Field(
        default_factory=lambda: os.getenv("WATSONX_BASE_URL", "https://us-south.ml.cloud.ibm.com"),
        description="A base url for accessing the watsonx.ai",
    )
    api_key: SecretStr | None = Field(
        default_factory=lambda: os.getenv("WATSONX_API_KEY"),
        description="The watsonx API key, only needed of using the hosted service",
    )
    project_id: str | None = Field(
        default_factory=lambda: os.getenv("WATSONX_PROJECT_ID"),
        description="The Project ID key, only needed of using the hosted service",
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
