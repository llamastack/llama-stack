# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from llama_stack.core.telemetry.trace_protocol import trace_protocol

from .model_schemas import (
    ListModelsResponse,
    Model,
    ModelType,
    OpenAIListModelsResponse,
)


@runtime_checkable
@trace_protocol
class ModelService(Protocol):
    async def list_models(self) -> ListModelsResponse:
        """List all models."""
        ...

    async def openai_list_models(self) -> OpenAIListModelsResponse:
        """List models using the OpenAI API."""
        ...

    async def get_model(
        self,
        model_id: str,
    ) -> Model:
        """Get model."""
        ...

    async def register_model(
        self,
        model_id: str,
        provider_model_id: str | None = None,
        provider_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        model_type: ModelType | None = None,
    ) -> Model:
        """Register model."""
        ...

    async def unregister_model(
        self,
        model_id: str,
    ) -> None:
        """Unregister model."""
        ...
