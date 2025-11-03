# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from llama_stack.apis.common.type_system import ParamType
from llama_stack.core.telemetry.trace_protocol import trace_protocol

from .models import ListScoringFunctionsResponse, ScoringFn, ScoringFnParams


@runtime_checkable
@trace_protocol
class ScoringFunctionsService(Protocol):
    async def list_scoring_functions(self) -> ListScoringFunctionsResponse:
        """List all scoring functions."""
        ...

    async def get_scoring_function(self, scoring_fn_id: str, /) -> ScoringFn:
        """Get a scoring function by its ID."""
        ...

    async def register_scoring_function(
        self,
        scoring_fn_id: str,
        description: str,
        return_type: ParamType,
        provider_scoring_fn_id: str | None = None,
        provider_id: str | None = None,
        params: ScoringFnParams | None = None,
    ) -> None:
        """Register a scoring function."""
        ...

    async def unregister_scoring_function(self, scoring_fn_id: str) -> None:
        """Unregister a scoring function."""
        ...
