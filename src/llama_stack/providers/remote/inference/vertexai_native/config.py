# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from llama_stack_api import json_schema_type


class VertexAINativeProviderDataValidator(BaseModel):
    """Validates optional request-scoped Vertex AI override fields."""

    vertex_project: str | None = Field(
        default=None,
        description="Google Cloud project ID for Vertex AI",
    )
    vertex_location: str | None = Field(
        default=None,
        description="Google Cloud location for Vertex AI (e.g., global)",
    )


@json_schema_type
class VertexAINativeConfig(RemoteInferenceProviderConfig):
    """Configuration for the Vertex AI native inference provider."""

    project: str = Field(
        description="Google Cloud project ID for Vertex AI",
    )
    location: str = Field(
        default="global",
        description="Google Cloud location for Vertex AI",
    )

    @classmethod
    def sample_run_config(
        cls,
        project: str = "${env.VERTEX_AI_PROJECT:=}",
        location: str = "${env.VERTEX_AI_LOCATION:=global}",
        **kwargs,
    ) -> dict[str, Any]:
        """Returns a template config with environment-variable placeholders."""

        return {
            "project": project,
            "location": location,
        }
