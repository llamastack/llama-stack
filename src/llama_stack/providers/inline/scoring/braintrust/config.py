# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any

from pydantic import BaseModel, Field


class BraintrustScoringConfig(BaseModel):
    openai_api_key: str | None = Field(
        default=None,
        description="The OpenAI API Key",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> dict[str, Any]:
        return {
            "openai_api_key": "${env.OPENAI_API_KEY:=}",
        }
