# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Datasets API protocol definition.

This module contains the Datasets protocol definition.
Pydantic models are defined in llama_stack_api.datasets.models.
The FastAPI router is defined in llama_stack_api.datasets.fastapi_routes.
"""

from typing import Protocol, runtime_checkable

from .models import (
    Dataset,
    GetDatasetRequest,
    ListDatasetsResponse,
    RegisterDatasetRequest,
    UnregisterDatasetRequest,
)


@runtime_checkable
class Datasets(Protocol):
    """Protocol for dataset management operations."""

    async def register_dataset(self, request: RegisterDatasetRequest) -> Dataset: ...

    async def get_dataset(self, request: GetDatasetRequest) -> Dataset: ...

    async def list_datasets(self) -> ListDatasetsResponse: ...

    async def unregister_dataset(self, request: UnregisterDatasetRequest) -> None: ...
