# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from .models import (
    GetScoringFunctionRequest,
    ListScoringFunctionsRequest,
    ListScoringFunctionsResponse,
    RegisterScoringFunctionRequest,
    ScoringFn,
    UnregisterScoringFunctionRequest,
)


@runtime_checkable
class ScoringFunctions(Protocol):
    async def list_scoring_functions(
        self,
        request: ListScoringFunctionsRequest,
    ) -> ListScoringFunctionsResponse: ...

    async def get_scoring_function(
        self,
        request: GetScoringFunctionRequest,
    ) -> ScoringFn: ...

    async def register_scoring_function(
        self,
        request: RegisterScoringFunctionRequest,
    ) -> None: ...

    async def unregister_scoring_function(
        self,
        request: UnregisterScoringFunctionRequest,
    ) -> None: ...
