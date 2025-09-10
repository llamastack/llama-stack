# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class DatabricksImplConfig(BaseModel):
    url: str = Field(
        default=None,
        description="The URL for the Databricks model serving endpoint",
    )
    api_token: str = Field(
        default=None,
        description="The Databricks API token",
    )

    @classmethod
    def sample_run_config(
        cls,
        url: str = "${env.DATABRICKS_URL:=}",
        api_token: str = "${env.DATABRICKS_API_TOKEN:=}",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "url": url,
            "api_token": api_token,
        }
