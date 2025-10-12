# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, SecretStr

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from llama_stack.schema_utils import json_schema_type


class AzureProviderDataValidator(BaseModel):
    azure_api_key: SecretStr = Field(
        description="Azure API key for Azure",
    )
    azure_api_base: HttpUrl = Field(
        description="Azure API base for Azure (e.g., https://your-resource-name.openai.azure.com)",
    )
    azure_api_version: str | None = Field(
        default=None,
        description="Azure API version for Azure (e.g., 2024-06-01)",
    )
    azure_api_type: str | None = Field(
        default="azure",
        description="Azure API type for Azure (e.g., azure)",
    )


@json_schema_type
class AzureConfig(RemoteInferenceProviderConfig):
    api_base: HttpUrl = Field(
        description="Azure API base for Azure (e.g., https://your-resource-name.openai.azure.com)",
    )
    api_version: str | None = Field(
        default_factory=lambda: os.getenv("AZURE_API_VERSION"),
        description="Azure API version for Azure (e.g., 2024-12-01-preview)",
    )
    api_type: str | None = Field(
        default_factory=lambda: os.getenv("AZURE_API_TYPE", "azure"),
        description="Azure API type for Azure (e.g., azure)",
    )

    @classmethod
    def sample_run_config(
        cls,
        api_key: str = "${env.AZURE_API_KEY:=}",
        api_base: str = "${env.AZURE_API_BASE:=}",
        api_version: str = "${env.AZURE_API_VERSION:=}",
        api_type: str = "${env.AZURE_API_TYPE:=}",
        **kwargs,
    ) -> dict[str, Any]:
        return {
            "api_key": api_key,
            "api_base": api_base,
            "api_version": api_version,
            "api_type": api_type,
        }
