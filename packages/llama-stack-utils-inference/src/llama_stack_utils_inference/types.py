# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import OpenAIAssistantMessageParam


class AssistantMessageWithReasoning(OpenAIAssistantMessageParam):
    """Internal type for passing reasoning content between the Responses
    layer and providers. NOT part of the public API.

    The Responses layer creates this when converting input ReasoningItems
    to CC messages. Providers check isinstance(msg, AssistantMessageWithReasoning)
    and map reasoning_content to their own CC format (e.g. 'reasoning' for Ollama/vLLM).
    """

    reasoning_content: str | None = None
