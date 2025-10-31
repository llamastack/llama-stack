# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from llama_stack.schema_utils import json_schema_type


class LlamaProviderDataValidator(BaseModel):
    llama_api_key: str | None = Field(
        default=None,
        description="API key for api.llama models",
    )


@json_schema_type
class LlamaCompatConfig(RemoteInferenceProviderConfig):
    openai_compat_api_base: str = Field(
        default="https://api.llama.com/compat/v1/",
        description="The URL for the Llama API server",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.LLAMA_API_KEY}", **kwargs) -> dict[str, Any]:
        return {
            "openai_compat_api_base": "https://api.llama.com/compat/v1/",
            "api_key": api_key,
        }
