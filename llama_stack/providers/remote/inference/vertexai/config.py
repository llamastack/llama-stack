# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, SecretStr

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from llama_stack.schema_utils import json_schema_type


class VertexAIProviderDataValidator(BaseModel):
    vertex_project: str | None = Field(
        default=None,
        description="Google Cloud project ID for Vertex AI",
    )
    vertex_location: str | None = Field(
        default=None,
        description="Google Cloud location for Vertex AI (e.g., us-central1)",
    )


@json_schema_type
class VertexAIConfig(RemoteInferenceProviderConfig):
    auth_credential: SecretStr | None = Field(default=None, exclude=True)

    project: str = Field(
        description="Google Cloud project ID for Vertex AI",
    )
    location: str = Field(
        default="us-central1",
        description="Google Cloud location for Vertex AI",
    )

    @classmethod
    def sample_run_config(
        cls,
        project: str = "${env.VERTEX_AI_PROJECT:=}",
        location: str = "${env.VERTEX_AI_LOCATION:=us-central1}",
        **kwargs,
    ) -> dict[str, Any]:
        return {
            "project": project,
            "location": location,
        }
