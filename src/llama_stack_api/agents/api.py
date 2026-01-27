# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from llama_stack_api.openai_responses import (
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
)

from .models import (
    CreateResponseRequest,
    DeleteResponseRequest,
    ListResponseInputItemsRequest,
    ListResponsesRequest,
    RetrieveResponseRequest,
)


@runtime_checkable
class Agents(Protocol):
    async def get_openai_response(
        self,
        request: RetrieveResponseRequest,
    ) -> OpenAIResponseObject: ...

    async def create_openai_response(
        self,
        request: CreateResponseRequest,
    ) -> OpenAIResponseObject | AsyncIterator[OpenAIResponseObjectStream]: ...

    async def list_openai_responses(
        self,
        request: ListResponsesRequest,
    ) -> ListOpenAIResponseObject: ...

    async def list_openai_response_input_items(
        self,
        request: ListResponseInputItemsRequest,
    ) -> ListOpenAIResponseInputItem: ...

    async def delete_openai_response(
        self,
        request: DeleteResponseRequest,
    ) -> OpenAIDeleteResponseObject: ...
