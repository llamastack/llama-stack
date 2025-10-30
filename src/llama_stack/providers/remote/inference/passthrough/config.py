# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack_api.schema_utils import json_schema_type
from pydantic import Field, SecretStr

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig


@json_schema_type
class PassthroughImplConfig(RemoteInferenceProviderConfig):
    url: str = Field(
        default=None,
        description="The URL for the passthrough endpoint",
    )

    api_key: SecretStr | None = Field(
        default=None,
        description="API Key for the passthrouth endpoint",
    )

    @classmethod
    def sample_run_config(
        cls, url: str = "${env.PASSTHROUGH_URL}", api_key: str = "${env.PASSTHROUGH_API_KEY}", **kwargs
    ) -> dict[str, Any]:
        return {
            "url": url,
            "api_key": api_key,
        }
