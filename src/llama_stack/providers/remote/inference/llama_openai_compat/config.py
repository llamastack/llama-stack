# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, HttpUrl

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from llama_stack_api import json_schema_type


class LlamaProviderDataValidator(BaseModel):
    llama_api_key: str | None = Field(
        default=None,
        description="API key for api.llama models",
    )


@json_schema_type
class LlamaCompatConfig(RemoteInferenceProviderConfig):
    base_url: HttpUrl | None = Field(
        default=HttpUrl("https://api.llama.com/compat/v1/"),
        description="The URL for the Llama API server",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.LLAMA_API_KEY}", **kwargs) -> dict[str, Any]:
        return {
            "base_url": "https://api.llama.com/compat/v1/",
            "api_key": api_key,
        }
