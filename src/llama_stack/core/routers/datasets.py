# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.log import get_logger
from llama_stack_api import (
    AppendRowsRequest,
    DatasetIO,
    DatasetPurpose,
    DataSource,
    IterRowsRequest,
    PaginatedResponse,
    RoutingTable,
)

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

    async def iterrows(self, request: IterRowsRequest) -> PaginatedResponse:
        logger.debug(
            f"DatasetIORouter.iterrows: {request.dataset_id}, start_index={request.start_index} limit={request.limit}",
        )
        provider = await self.routing_table.get_provider_impl(request.dataset_id)
        return await provider.iterrows(
            dataset_id=request.dataset_id,
            start_index=request.start_index,
            limit=request.limit,
        )

    async def append_rows(self, request: AppendRowsRequest) -> None:
        logger.debug(f"DatasetIORouter.append_rows: {request.dataset_id}, {len(request.rows)} rows")
        provider = await self.routing_table.get_provider_impl(request.dataset_id)
        return await provider.append_rows(
            dataset_id=request.dataset_id,
            rows=request.rows,
        )
