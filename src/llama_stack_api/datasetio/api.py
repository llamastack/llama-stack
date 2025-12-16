# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""DatasetIO API protocol definition.

This module contains the DatasetIO protocol definition.
Pydantic models are defined in llama_stack_api.datasetio.models.
The FastAPI router is defined in llama_stack_api.datasetio.fastapi_routes.
"""

from typing import Protocol, runtime_checkable

from llama_stack_api.datasets import Dataset

from .models import (
    AppendRowsRequest,
    IterRowsRequest,
    PaginatedResponse,
)


class DatasetStore(Protocol):
    def get_dataset(self, dataset_id: str) -> Dataset: ...


@runtime_checkable
class DatasetIO(Protocol):
    """Protocol for dataset I/O operations.

    The DatasetIO API provides operations for reading and writing data to datasets.
    This includes iterating over rows and appending new rows to existing datasets.
    """

    # keeping for aligning with inference/safety, but this is not used
    dataset_store: DatasetStore

    async def iterrows(self, request: IterRowsRequest) -> PaginatedResponse: ...

    async def append_rows(self, request: AppendRowsRequest) -> None: ...
