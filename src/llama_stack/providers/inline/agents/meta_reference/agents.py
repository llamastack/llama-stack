# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator

from llama_stack.core.datatypes import AccessRule
from llama_stack.core.storage.kvstore import InmemoryKVStoreImpl, kvstore_impl
from llama_stack.log import get_logger
from llama_stack.providers.utils.responses.responses_store import ResponsesStore
from llama_stack_api import (
    Agents,
    Connectors,
    Conversations,
    CreateResponseRequest,
    DeleteResponseRequest,
    Files,
    Inference,
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    ListResponseInputItemsRequest,
    ListResponsesRequest,
    OpenAIDeleteResponseObject,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    Prompts,
    RetrieveResponseRequest,
    Safety,
    ToolGroups,
    ToolRuntime,
    VectorIO,
)

from .config import MetaReferenceAgentsImplConfig
from .responses.openai_responses import OpenAIResponsesImpl

logger = get_logger(name=__name__, category="agents::meta_reference")


class MetaReferenceAgentsImpl(Agents):
    def __init__(
        self,
        config: MetaReferenceAgentsImplConfig,
        inference_api: Inference,
        vector_io_api: VectorIO,
        safety_api: Safety | None,
        tool_runtime_api: ToolRuntime,
        tool_groups_api: ToolGroups,
        conversations_api: Conversations,
        prompts_api: Prompts,
        files_api: Files,
        connectors_api: Connectors,
        policy: list[AccessRule],
    ):
        self.config = config
        self.inference_api = inference_api
        self.vector_io_api = vector_io_api
        self.safety_api = safety_api
        self.tool_runtime_api = tool_runtime_api
        self.tool_groups_api = tool_groups_api
        self.conversations_api = conversations_api
        self.prompts_api = prompts_api
        self.files_api = files_api
        self.in_memory_store = InmemoryKVStoreImpl()
        self.openai_responses_impl: OpenAIResponsesImpl | None = None
        self.policy = policy
        self.connectors_api = connectors_api

    async def initialize(self) -> None:
        self.persistence_store = await kvstore_impl(self.config.persistence.agent_state)
        self.responses_store = ResponsesStore(self.config.persistence.responses, self.policy)
        await self.responses_store.initialize()
        self.openai_responses_impl = OpenAIResponsesImpl(
            inference_api=self.inference_api,
            tool_groups_api=self.tool_groups_api,
            tool_runtime_api=self.tool_runtime_api,
            responses_store=self.responses_store,
            vector_io_api=self.vector_io_api,
            safety_api=self.safety_api,
            conversations_api=self.conversations_api,
            prompts_api=self.prompts_api,
            files_api=self.files_api,
            vector_stores_config=self.config.vector_stores_config,
            connectors_api=self.connectors_api,
        )

    async def shutdown(self) -> None:
        pass

    # OpenAI responses
    async def get_openai_response(
        self,
        request: RetrieveResponseRequest,
    ) -> OpenAIResponseObject:
        assert self.openai_responses_impl is not None, "OpenAI responses not initialized"
        return await self.openai_responses_impl.get_openai_response(request.response_id)

    async def create_openai_response(
        self,
        request: CreateResponseRequest,
    ) -> OpenAIResponseObject | AsyncIterator[OpenAIResponseObjectStream]:
        """Create an OpenAI response.

        Returns either a single response object (non-streaming) or an async iterator
        yielding response stream events (streaming).
        """
        assert self.openai_responses_impl is not None, "OpenAI responses not initialized"
        result = await self.openai_responses_impl.create_openai_response(
            request.input,
            request.model,
            request.prompt,
            request.instructions,
            request.previous_response_id,
            request.conversation,
            request.store,
            request.stream,
            request.temperature,
            request.text,
            request.tool_choice,
            request.tools,
            request.include,
            request.max_infer_iters,
            request.guardrails,
            request.parallel_tool_calls,
            request.max_tool_calls,
            request.max_output_tokens,
            request.reasoning,
            request.metadata,
        )
        return result

    async def list_openai_responses(
        self,
        request: ListResponsesRequest,
    ) -> ListOpenAIResponseObject:
        assert self.openai_responses_impl is not None, "OpenAI responses not initialized"
        return await self.openai_responses_impl.list_openai_responses(
            request.after, request.limit, request.model, request.order
        )

    async def list_openai_response_input_items(
        self,
        request: ListResponseInputItemsRequest,
    ) -> ListOpenAIResponseInputItem:
        assert self.openai_responses_impl is not None, "OpenAI responses not initialized"
        return await self.openai_responses_impl.list_openai_response_input_items(
            request.response_id,
            request.after,
            request.before,
            request.include,
            request.limit,
            request.order,
        )

    async def delete_openai_response(
        self,
        request: DeleteResponseRequest,
    ) -> OpenAIDeleteResponseObject:
        assert self.openai_responses_impl is not None, "OpenAI responses not initialized"
        return await self.openai_responses_impl.delete_openai_response(request.response_id)
