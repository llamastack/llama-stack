# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator

from llama_stack.apis.agents import (
    Agents,
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIResponseInput,
    OpenAIResponseInputTool,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    Order,
)
from llama_stack.apis.agents.agents import ResponseGuardrail
from llama_stack.apis.agents.openai_responses import OpenAIResponsePrompt, OpenAIResponseText
from llama_stack.apis.conversations import Conversations
from llama_stack.apis.inference import Inference
from llama_stack.apis.safety import Safety
from llama_stack.apis.tools import ToolGroups, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.core.datatypes import AccessRule
from llama_stack.log import get_logger

from .config import DanaAgentsImplConfig

logger = get_logger(name=__name__, category="agents::dana")


class DanaAgentsImpl(Agents):
    """Stub implementation of the Agents API using the Dana library."""

    def __init__(
        self,
        config: DanaAgentsImplConfig,
        inference_api: Inference,
        vector_io_api: VectorIO,
        safety_api: Safety,
        tool_runtime_api: ToolRuntime,
        tool_groups_api: ToolGroups,
        conversations_api: Conversations,
        policy: list[AccessRule],
        telemetry_enabled: bool = False,
    ):
        self.config = config
        self.inference_api = inference_api
        self.vector_io_api = vector_io_api
        self.safety_api = safety_api
        self.tool_runtime_api = tool_runtime_api
        self.tool_groups_api = tool_groups_api
        self.conversations_api = conversations_api
        self.telemetry_enabled = telemetry_enabled
        self.policy = policy

    async def initialize(self) -> None:
        """Initialize the Dana agents implementation."""
        # TODO: Initialize Dana library here
        logger.info("Dana agents implementation initialized (stub)")

    async def shutdown(self) -> None:
        """Shutdown the Dana agents implementation."""
        # TODO: Cleanup Dana library here
        pass

    # OpenAI responses
    async def get_openai_response(
        self,
        response_id: str,
    ) -> OpenAIResponseObject:
        """Get a model response."""
        raise NotImplementedError("Dana provider is not yet implemented")

    async def create_openai_response(
        self,
        input: str | list[OpenAIResponseInput],
        model: str,
        prompt: OpenAIResponsePrompt | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        conversation: str | None = None,
        store: bool | None = True,
        stream: bool | None = False,
        temperature: float | None = None,
        text: OpenAIResponseText | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
        include: list[str] | None = None,
        max_infer_iters: int | None = 10,
        guardrails: list[ResponseGuardrail] | None = None,
    ) -> OpenAIResponseObject | AsyncIterator[OpenAIResponseObjectStream]:
        """Create a model response."""
        raise NotImplementedError("Dana provider is not yet implemented")

    async def list_openai_responses(
        self,
        after: str | None = None,
        limit: int | None = 50,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseObject:
        """List all responses."""
        raise NotImplementedError("Dana provider is not yet implemented")

    async def list_openai_response_input_items(
        self,
        response_id: str,
        after: str | None = None,
        before: str | None = None,
        include: list[str] | None = None,
        limit: int | None = 20,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseInputItem:
        """List input items."""
        raise NotImplementedError("Dana provider is not yet implemented")

    async def delete_openai_response(self, response_id: str) -> OpenAIDeleteResponseObject:
        """Delete a response."""
        raise NotImplementedError("Dana provider is not yet implemented")
