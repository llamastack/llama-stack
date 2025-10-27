# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, SecretStr

from llama_stack.schema_utils import json_schema_type


class SambaNovaProviderDataValidator(BaseModel):
    sambanova_api_key: str | None = Field(
        default=None,
        description="Sambanova Cloud API key",
    )


@json_schema_type
class SambaNovaSafetyConfig(BaseModel):
    url: str = Field(
        default="https://api.sambanova.ai/v1",
        description="The URL for the SambaNova AI server",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="The SambaNova cloud API Key",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.SAMBANOVA_API_KEY:=}", **kwargs) -> dict[str, Any]:
        return {
            "url": "https://api.sambanova.ai/v1",
            "api_key": api_key,
        }
