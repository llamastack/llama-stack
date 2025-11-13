# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack_api import json_schema_type
from pydantic import Field

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig


@json_schema_type
class FireworksImplConfig(RemoteInferenceProviderConfig):
    url: str = Field(
        default="https://api.fireworks.ai/inference/v1",
        description="The URL for the Fireworks server",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.FIREWORKS_API_KEY:=}", **kwargs) -> dict[str, Any]:
        return {
            "url": "https://api.fireworks.ai/inference/v1",
            "api_key": api_key,
        }
