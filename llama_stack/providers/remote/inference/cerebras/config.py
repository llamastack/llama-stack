# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Any

from pydantic import BaseModel, Field, SecretStr

from llama_stack.schema_utils import json_schema_type

DEFAULT_BASE_URL = "https://api.cerebras.ai"


@json_schema_type
class CerebrasImplConfig(BaseModel):
    base_url: str = Field(
        default=os.environ.get("CEREBRAS_BASE_URL", DEFAULT_BASE_URL),
        description="Base URL for the Cerebras API",
    )
    api_key: SecretStr | None = Field(
        default=os.environ.get("CEREBRAS_API_KEY"),
        description="Cerebras API Key",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.CEREBRAS_API_KEY:=}", **kwargs) -> dict[str, Any]:
        return {
            "base_url": DEFAULT_BASE_URL,
            "api_key": api_key,
        }
