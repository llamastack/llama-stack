# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.core.datatypes import AccessRule
from llama_stack.core.storage.kvstore import InmemoryKVStoreImpl, kvstore_impl
from llama_stack.log import get_logger
from llama_stack.providers.utils.responses.responses_store import ResponsesStore
from llama_stack_api import (
    Agents,
    Conversations,
    Files,
    Inference,
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIResponseInput,
    OpenAIResponseInputTool,
    OpenAIResponseInputToolChoice,
    OpenAIResponseObject,
    OpenAIResponsePrompt,
    OpenAIResponseText,
    Order,
    Prompts,
    ResponseGuardrail,
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
        )

    async def shutdown(self) -> None:
        pass

    # OpenAI responses
    async def get_openai_response(
        self,
        response_id: str,
    ) -> OpenAIResponseObject:
        assert self.openai_responses_impl is not None, "OpenAI responses not initialized"
        return await self.openai_responses_impl.get_openai_response(response_id)

    async def create_openai_response(
        self,
        input: str | list[OpenAIResponseInput],
        model: str,
        prompt: OpenAIResponsePrompt | None = None,
        instructions: str | None = None,
        parallel_tool_calls: bool | None = True,
        previous_response_id: str | None = None,
        conversation: str | None = None,
        store: bool | None = True,
        stream: bool | None = False,
        temperature: float | None = None,
        text: OpenAIResponseText | None = None,
        tool_choice: OpenAIResponseInputToolChoice | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
        include: list[str] | None = None,
        max_infer_iters: int | None = 10,
        guardrails: list[ResponseGuardrail] | None = None,
        max_tool_calls: int | None = None,
        metadata: dict[str, str] | None = None,
    ) -> OpenAIResponseObject:
        assert self.openai_responses_impl is not None, "OpenAI responses not initialized"
        result = await self.openai_responses_impl.create_openai_response(
            input,
            model,
            prompt,
            instructions,
            previous_response_id,
            conversation,
            store,
            stream,
            temperature,
            text,
            tool_choice,
            tools,
            include,
            max_infer_iters,
            guardrails,
            parallel_tool_calls,
            max_tool_calls,
            metadata,
        )
        return result  # type: ignore[no-any-return]

    async def list_openai_responses(
        self,
        after: str | None = None,
        limit: int | None = 50,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseObject:
        assert self.openai_responses_impl is not None, "OpenAI responses not initialized"
        return await self.openai_responses_impl.list_openai_responses(after, limit, model, order)

    async def list_openai_response_input_items(
        self,
        response_id: str,
        after: str | None = None,
        before: str | None = None,
        include: list[str] | None = None,
        limit: int | None = 20,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseInputItem:
        assert self.openai_responses_impl is not None, "OpenAI responses not initialized"
        return await self.openai_responses_impl.list_openai_response_input_items(
            response_id, after, before, include, limit, order
        )

    async def delete_openai_response(self, response_id: str) -> OpenAIDeleteResponseObject:
        assert self.openai_responses_impl is not None, "OpenAI responses not initialized"
        return await self.openai_responses_impl.delete_openai_response(response_id)
