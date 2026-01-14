# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.core.datatypes import (
    ScoringFnWithOwner,
)
from llama_stack.log import get_logger
from llama_stack_api import (
    GetScoringFunctionRequest,
    ListScoringFunctionsRequest,
    ListScoringFunctionsResponse,
    RegisterScoringFunctionRequest,
    ResourceType,
    ScoringFn,
    ScoringFunctions,
    UnregisterScoringFunctionRequest,
)

from .common import CommonRoutingTableImpl

logger = get_logger(name=__name__, category="core::routing_tables")


class ScoringFunctionsRoutingTable(CommonRoutingTableImpl, ScoringFunctions):
    async def list_scoring_functions(self, request: ListScoringFunctionsRequest) -> ListScoringFunctionsResponse:
        return ListScoringFunctionsResponse(data=await self.get_all_with_type(ResourceType.scoring_function.value))

    async def get_scoring_function(self, request: GetScoringFunctionRequest) -> ScoringFn:
        scoring_fn = await self.get_object_by_identifier("scoring_function", request.scoring_fn_id)
        if scoring_fn is None:
            raise ValueError(f"Scoring function '{request.scoring_fn_id}' not found")
        return scoring_fn

    async def register_scoring_function(
        self,
        request: RegisterScoringFunctionRequest,
    ) -> None:
        provider_scoring_fn_id = request.provider_scoring_fn_id
        if provider_scoring_fn_id is None:
            provider_scoring_fn_id = request.scoring_fn_id
        provider_id = request.provider_id
        if provider_id is None:
            if len(self.impls_by_provider_id) == 1:
                provider_id = list(self.impls_by_provider_id.keys())[0]
            else:
                raise ValueError(
                    "No provider specified and multiple providers available. Please specify a provider_id."
                )
        scoring_fn = ScoringFnWithOwner(
            identifier=request.scoring_fn_id,
            description=request.description,
            return_type=request.return_type,
            provider_resource_id=provider_scoring_fn_id,
            provider_id=provider_id,
            params=request.params,
        )
        scoring_fn.provider_id = provider_id
        await self.register_object(scoring_fn)

    async def unregister_scoring_function(self, request: UnregisterScoringFunctionRequest) -> None:
        get_request = GetScoringFunctionRequest(scoring_fn_id=request.scoring_fn_id)
        existing_scoring_fn = await self.get_scoring_function(get_request)
        await self.unregister_object(existing_scoring_fn)
