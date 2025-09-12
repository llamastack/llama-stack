# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from llama_stack.apis.common.job_types import Job
from llama_stack.apis.inference import (
    InterleavedContent,
    LogProbConfig,
    Message,
    ResponseFormat,
    SamplingParams,
    ToolChoice,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.schema_utils import webmethod


@runtime_checkable
class BatchInference(Protocol):
    """Batch inference API for generating completions and chat completions.

    This is an asynchronous API. If the request is successful, the response will be a job which can be polled for completion.

    NOTE: This API is not yet implemented and is subject to change in concert with other asynchronous APIs
    including (post-training, evals, etc).
    """

    @webmethod(route="/batch-inference/completion", method="POST", level=LLAMA_STACK_API_V1)
    async def completion(
        self,
        model: str,
        content_batch: list[InterleavedContent],
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        logprobs: LogProbConfig | None = None,
    ) -> Job:
        """Generate completions for a batch of content.

        :param model: The model to use for the completion.
        :param content_batch: The content to complete.
        :param sampling_params: The sampling parameters to use for the completion.
        :param response_format: The response format to use for the completion.
        :param logprobs: The logprobs to use for the completion.
        :returns: A job for the completion.
        """
        ...

    @webmethod(route="/batch-inference/chat-completion", method="POST", level=LLAMA_STACK_API_V1)
    async def chat_completion(
        self,
        model: str,
        messages_batch: list[list[Message]],
        sampling_params: SamplingParams | None = None,
        # zero-shot tool definitions as input to the model
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = ToolChoice.auto,
        tool_prompt_format: ToolPromptFormat | None = None,
        response_format: ResponseFormat | None = None,
        logprobs: LogProbConfig | None = None,
    ) -> Job:
        """Generate chat completions for a batch of messages.

        :param model: The model to use for the chat completion.
        :param messages_batch: The messages to complete.
        :param sampling_params: The sampling parameters to use for the completion.
        :param tools: The tools to use for the chat completion.
        :param tool_choice: The tool choice to use for the chat completion.
        :param tool_prompt_format: The tool prompt format to use for the chat completion.
        :param response_format: The response format to use for the chat completion.
        :param logprobs: The logprobs to use for the chat completion.
        :returns: A job for the chat completion.
        """
        ...
