# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, SecretStr

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from llama_stack_api import json_schema_type


class VertexAIProviderDataValidator(BaseModel):
    vertex_project: str | None = Field(
        default=None,
        description="Google Cloud project ID for Vertex AI",
    )
    vertex_location: str | None = Field(
        default=None,
        description="Google Cloud location for Vertex AI (e.g., global)",
    )
    vertex_access_token: str | None = None


@json_schema_type
class VertexAIConfig(RemoteInferenceProviderConfig):
    auth_credential: SecretStr | None = Field(
        default=None,
        exclude=True,
        description="Deprecated: has no effect. Use access_token or ADC instead.",
    )

    project: str = Field(
        description="Google Cloud project ID for Vertex AI",
    )
    location: str = Field(
        default="global",
        description="Google Cloud location for Vertex AI",
    )
    access_token: SecretStr | None = Field(
        default=None,
        description="Optional access token for Vertex AI authentication. When set, this is used instead of Application Default Credentials (ADC).",
        json_schema_extra={"env": "VERTEX_AI_ACCESS_TOKEN"},
    )

    @classmethod
    def sample_run_config(
        cls,
        project: str = "${env.VERTEX_AI_PROJECT:=}",
        location: str = "${env.VERTEX_AI_LOCATION:=global}",
        **kwargs,
    ) -> dict[str, Any]:
        return {
            "project": project,
            "location": location,
        }
