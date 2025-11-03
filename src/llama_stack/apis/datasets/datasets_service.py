# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from llama_stack.core.telemetry.trace_protocol import trace_protocol

from .models import Dataset, DatasetPurpose, DataSource, ListDatasetsResponse


@runtime_checkable
@trace_protocol
class DatasetsService(Protocol):
    async def register_dataset(
        self,
        purpose: DatasetPurpose,
        source: DataSource,
        metadata: dict[str, Any] | None = None,
        dataset_id: str | None = None,
    ) -> Dataset:
        """
        Register a new dataset.

        :param purpose: The purpose of the dataset.
        One of:
            - "post-training/messages": The dataset contains a messages column with list of messages for post-training.
            - "eval/question-answer": The dataset contains a question column and an answer column for evaluation.
            - "eval/messages-answer": The dataset contains a messages column with list of messages and an answer column for evaluation.
        :param source: The data source of the dataset. Ensure that the data source schema is compatible with the purpose of the dataset.
        :param metadata: The metadata for the dataset.
        :param dataset_id: The ID of the dataset. If not provided, an ID will be generated.
        :returns: A Dataset.
        """
        ...

    async def get_dataset(
        self,
        dataset_id: str,
    ) -> Dataset:
        """Get a dataset by its ID.

        :param dataset_id: The ID of the dataset to get.
        :returns: A Dataset.
        """
        ...

    async def list_datasets(self) -> ListDatasetsResponse:
        """List all datasets.

        :returns: A ListDatasetsResponse.
        """
        ...

    async def unregister_dataset(
        self,
        dataset_id: str,
    ) -> None:
        """Unregister a dataset by its ID.

        :param dataset_id: The ID of the dataset to unregister.
        """
        ...
