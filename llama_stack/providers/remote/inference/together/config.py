# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import Field

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class TogetherImplConfig(RemoteInferenceProviderConfig):
    url: str = Field(
        default="https://api.together.xyz/v1",
        description="The URL for the Together AI server",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> dict[str, Any]:
        return {
            "url": "https://api.together.xyz/v1",
            "api_key": "${env.TOGETHER_API_KEY:=}",
        }
