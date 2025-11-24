# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import Field, HttpUrl

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from llama_stack_api import json_schema_type


@json_schema_type
class PassthroughImplConfig(RemoteInferenceProviderConfig):
    base_url: HttpUrl | None = Field(
        default=None,
        description="The URL for the passthrough endpoint",
    )

    @classmethod
    def sample_run_config(
        cls, base_url: HttpUrl | None = "${env.PASSTHROUGH_URL}", api_key: str = "${env.PASSTHROUGH_API_KEY}", **kwargs
    ) -> dict[str, Any]:
        return {
            "base_url": base_url,
            "api_key": api_key,
        }
