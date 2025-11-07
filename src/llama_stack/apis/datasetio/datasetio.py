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
class DatasetIO(Protocol):
    # keeping for aligning with inference/safety, but this is not used
    dataset_store: DatasetStore

    @webmethod(route="/datasetio/iterrows/{dataset_id:path}", method="GET", level=LLAMA_STACK_API_V1BETA)
    async def iterrows(
        self,
        dataset_id: str,
        start_index: int | None = None,
        limit: int | None = None,
    ) -> PaginatedResponse:
        """Get a paginated list of rows from a dataset.

        Uses offset-based pagination where:
        - start_index: The starting index (0-based). If None, starts from beginning.
        - limit: Number of items to return. If None or -1, returns all items.

        The response includes:
        - data: List of items for the current page.
        - has_more: Whether there are more items available after this set.

        :param dataset_id: The ID of the dataset to get the rows from.
        :param start_index: Index into dataset for the first row to get. Get all rows if None.
        :param limit: The number of rows to get.
        :returns: A PaginatedResponse.
        """
        ...

    @webmethod(route="/datasetio/append-rows/{dataset_id:path}", method="POST", level=LLAMA_STACK_API_V1BETA)
    async def append_rows(self, dataset_id: str, rows: list[dict[str, Any]]) -> None:
        """Append rows to a dataset.

        :param dataset_id: The ID of the dataset to append the rows to.
        :param rows: The rows to append to the dataset.
        """
        ...
