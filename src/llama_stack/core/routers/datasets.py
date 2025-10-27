# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.common.responses import PaginatedResponse
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import DatasetPurpose, DataSource
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import RoutingTable

logger = get_logger(name=__name__, category="core::routers")


class DatasetIORouter(DatasetIO):
    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        logger.debug("Initializing DatasetIORouter")
        self.routing_table = routing_table

    async def initialize(self) -> None:
        logger.debug("DatasetIORouter.initialize")
        pass

    async def shutdown(self) -> None:
        logger.debug("DatasetIORouter.shutdown")
        pass

    async def register_dataset(
        self,
        purpose: DatasetPurpose,
        source: DataSource,
        metadata: dict[str, Any] | None = None,
        dataset_id: str | None = None,
    ) -> None:
        logger.debug(
            f"DatasetIORouter.register_dataset: {purpose=} {source=} {metadata=} {dataset_id=}",
        )
        await self.routing_table.register_dataset(
            purpose=purpose,
            source=source,
            metadata=metadata,
            dataset_id=dataset_id,
        )

    async def iterrows(
        self,
        dataset_id: str,
        start_index: int | None = None,
        limit: int | None = None,
    ) -> PaginatedResponse:
        logger.debug(
            f"DatasetIORouter.iterrows: {dataset_id}, {start_index=} {limit=}",
        )
        provider = await self.routing_table.get_provider_impl(dataset_id)
        return await provider.iterrows(
            dataset_id=dataset_id,
            start_index=start_index,
            limit=limit,
        )

    async def append_rows(self, dataset_id: str, rows: list[dict[str, Any]]) -> None:
        logger.debug(f"DatasetIORouter.append_rows: {dataset_id}, {len(rows)} rows")
        provider = await self.routing_table.get_provider_impl(dataset_id)
        return await provider.append_rows(
            dataset_id=dataset_id,
            rows=rows,
        )
