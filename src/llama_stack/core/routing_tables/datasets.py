# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid

from llama_stack.core.datatypes import (
    DatasetWithOwner,
)
from llama_stack.log import get_logger
from llama_stack_api import (
    Dataset,
    DatasetNotFoundError,
    DatasetType,
    ListDatasetsResponse,
    ResourceType,
    RowsDataSource,
    URIDataSource,
)
from llama_stack_api.datasets.api import (
    Datasets,
    GetDatasetRequest,
    RegisterDatasetRequest,
    UnregisterDatasetRequest,
)

from .common import CommonRoutingTableImpl

logger = get_logger(name=__name__, category="core::routing_tables")


class DatasetsRoutingTable(CommonRoutingTableImpl, Datasets):
    async def list_datasets(self) -> ListDatasetsResponse:
        return ListDatasetsResponse(data=await self.get_all_with_type(ResourceType.dataset.value))

    async def get_dataset(self, request: GetDatasetRequest) -> Dataset:
        dataset = await self.get_object_by_identifier("dataset", request.dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(request.dataset_id)
        return dataset

    async def register_dataset(self, request: RegisterDatasetRequest) -> Dataset:
        purpose = request.purpose
        source = request.source
        metadata = request.metadata
        dataset_id = request.dataset_id
        if isinstance(source, dict):
            if source["type"] == "uri":
                source = URIDataSource.parse_obj(source)
            elif source["type"] == "rows":
                source = RowsDataSource.parse_obj(source)

        if not dataset_id:
            dataset_id = f"dataset-{str(uuid.uuid4())}"

        provider_dataset_id = dataset_id

        # infer provider from source
        if metadata and metadata.get("provider_id"):
            provider_id = metadata.get("provider_id")  # pass through from nvidia datasetio
        elif source.type == DatasetType.rows.value:
            provider_id = "localfs"
        elif source.type == DatasetType.uri.value:
            # infer provider from uri
            if source.uri.startswith("huggingface"):
                provider_id = "huggingface"
            else:
                provider_id = "localfs"
        else:
            raise ValueError(f"Unknown data source type: {source.type}")

        if metadata is None:
            metadata = {}

        dataset = DatasetWithOwner(
            identifier=dataset_id,
            provider_resource_id=provider_dataset_id,
            provider_id=provider_id,
            purpose=purpose,
            source=source,
            metadata=metadata,
        )

        await self.register_object(dataset)
        return dataset

    async def unregister_dataset(self, request: UnregisterDatasetRequest) -> None:
        dataset = await self.get_dataset(GetDatasetRequest(dataset_id=request.dataset_id))
        await self.unregister_object(dataset)
