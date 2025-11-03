# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from llama_stack.apis.common.responses import PaginatedResponse
from llama_stack.apis.datasets import Dataset
from llama_stack.apis.version import LLAMA_STACK_API_V1BETA
from llama_stack.schema_utils import webmethod


class DatasetStore(Protocol):
    def get_dataset(self, dataset_id: str) -> Dataset: ...


@runtime_checkable
class DatasetIOService(Protocol):
    # keeping for aligning with inference/safety, but this is not used
    dataset_store: DatasetStore

    @webmethod(route="/datasetio/iterrows/{dataset_id:path}", method="GET", level=LLAMA_STACK_API_V1BETA)
    async def iterrows(
        self,
        dataset_id: str,
        start_index: int | None = None,
        limit: int | None = None,
    ) -> PaginatedResponse:
        """Get a paginated list of rows from a dataset."""
        ...

    @webmethod(route="/datasetio/append-rows/{dataset_id:path}", method="POST", level=LLAMA_STACK_API_V1BETA)
    async def append_rows(self, dataset_id: str, rows: list[dict[str, Any]]) -> None:
        """Append rows to a dataset."""
        ...
