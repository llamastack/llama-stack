# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, SecretStr

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from llama_stack.schema_utils import json_schema_type


class DatabricksProviderDataValidator(BaseModel):
    databricks_api_token: str | None = Field(
        default=None,
        description="API token for Databricks models",
    )


@json_schema_type
class DatabricksImplConfig(RemoteInferenceProviderConfig):
    url: str | None = Field(
        default=None,
        description="The URL for the Databricks model serving endpoint",
    )
    auth_credential: SecretStr | None = Field(
        default=None,
        alias="api_token",
        description="The Databricks API token",
    )

    @classmethod
    def sample_run_config(
        cls,
        url: str = "${env.DATABRICKS_HOST:=}",
        api_token: str = "${env.DATABRICKS_TOKEN:=}",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "url": url,
            "api_token": api_token,
        }
