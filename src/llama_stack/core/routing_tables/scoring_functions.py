# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.common.type_system import ParamType
from llama_stack.apis.resource import ResourceType
from llama_stack.apis.scoring_functions import (
    ListScoringFunctionsResponse,
    ScoringFn,
    ScoringFnParams,
    ScoringFunctions,
)
from llama_stack.core.datatypes import (
    ScoringFnWithOwner,
)
from llama_stack.log import get_logger

from .common import CommonRoutingTableImpl

logger = get_logger(name=__name__, category="core::routing_tables")


class ScoringFunctionsRoutingTable(CommonRoutingTableImpl, ScoringFunctions):
    async def list_scoring_functions(self) -> ListScoringFunctionsResponse:
        return ListScoringFunctionsResponse(data=await self.get_all_with_type(ResourceType.scoring_function.value))

    async def get_scoring_function(self, scoring_fn_id: str) -> ScoringFn:
        scoring_fn = await self.get_object_by_identifier("scoring_function", scoring_fn_id)
        if scoring_fn is None:
            raise ValueError(f"Scoring function '{scoring_fn_id}' not found")
        return scoring_fn

    async def register_scoring_function(
        self,
        scoring_fn_id: str,
        description: str,
        return_type: ParamType,
        provider_scoring_fn_id: str | None = None,
        provider_id: str | None = None,
        params: ScoringFnParams | None = None,
    ) -> None:
        if provider_scoring_fn_id is None:
            provider_scoring_fn_id = scoring_fn_id
        if provider_id is None:
            if len(self.impls_by_provider_id) == 1:
                provider_id = list(self.impls_by_provider_id.keys())[0]
            else:
                raise ValueError(
                    "No provider specified and multiple providers available. Please specify a provider_id."
                )
        scoring_fn = ScoringFnWithOwner(
            identifier=scoring_fn_id,
            description=description,
            return_type=return_type,
            provider_resource_id=provider_scoring_fn_id,
            provider_id=provider_id,
            params=params,
        )
        scoring_fn.provider_id = provider_id
        await self.register_object(scoring_fn)

    async def unregister_scoring_function(self, scoring_fn_id: str) -> None:
        existing_scoring_fn = await self.get_scoring_function(scoring_fn_id)
        await self.unregister_object(existing_scoring_fn)
