# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


class GroqProviderDataValidator(BaseModel):
    groq_api_key: str | None = Field(
        default=None,
        description="API key for Groq models",
    )


@json_schema_type
class GroqConfig(BaseModel):
    api_key: str | None = Field(
        # The Groq client library loads the GROQ_API_KEY environment variable by default
        default=None,
        description="The Groq API key",
    )

    url: str = Field(
        default="https://api.groq.com",
        description="The URL for the Groq AI server",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.GROQ_API_KEY:=}", **kwargs) -> dict[str, Any]:
        return {
            "url": "https://api.groq.com",
            "api_key": api_key,
        }
