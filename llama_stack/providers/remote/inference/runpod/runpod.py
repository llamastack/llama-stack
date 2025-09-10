# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from collections.abc import AsyncGenerator

from openai import OpenAI

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.inference import OpenAIEmbeddingsResponse

# from llama_stack.providers.datatypes import ModelsProtocolPrivate
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper, build_hf_repo_model_entry
from llama_stack.providers.utils.inference.openai_compat import (
    OpenAIChatCompletionToLlamaStackMixin,
    OpenAICompletionToLlamaStackMixin,
    get_sampling_options,
    process_chat_completion_response,
    process_chat_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
)

from .config import RunpodImplConfig

# https://docs.runpod.io/serverless/vllm/overview#compatible-models
# https://github.com/runpod-workers/worker-vllm/blob/main/README.md#compatible-model-architectures
RUNPOD_SUPPORTED_MODELS = {
    "Llama3.1-8B": "meta-llama/Llama-3.1-8B",
    "Llama3.1-70B": "meta-llama/Llama-3.1-70B",
    "Llama3.1-405B:bf16-mp8": "meta-llama/Llama-3.1-405B",
    "Llama3.1-405B": "meta-llama/Llama-3.1-405B-FP8",
    "Llama3.1-405B:bf16-mp16": "meta-llama/Llama-3.1-405B",
    "Llama3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama3.1-70B-Instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "Llama3.1-405B-Instruct:bf16-mp8": "meta-llama/Llama-3.1-405B-Instruct",
    "Llama3.1-405B-Instruct": "meta-llama/Llama-3.1-405B-Instruct-FP8",
    "Llama3.1-405B-Instruct:bf16-mp16": "meta-llama/Llama-3.1-405B-Instruct",
    "Llama3.2-1B": "meta-llama/Llama-3.2-1B",
    "Llama3.2-3B": "meta-llama/Llama-3.2-3B",
}

SAFETY_MODELS_ENTRIES = []

# Create MODEL_ENTRIES from RUNPOD_SUPPORTED_MODELS for compatibility with starter template
MODEL_ENTRIES = [
    build_hf_repo_model_entry(provider_model_id, model_descriptor)
    for provider_model_id, model_descriptor in RUNPOD_SUPPORTED_MODELS.items()
] + SAFETY_MODELS_ENTRIES


class RunpodInferenceAdapter(
    ModelRegistryHelper,
    Inference,
    OpenAIChatCompletionToLlamaStackMixin,
    OpenAICompletionToLlamaStackMixin,
):
    def __init__(self, config: RunpodImplConfig) -> None:
        ModelRegistryHelper.__init__(self, stack_to_provider_models_map=RUNPOD_SUPPORTED_MODELS)
        self.config = config

    async def initialize(self) -> None:
        return

    async def shutdown(self) -> None:
        pass

    async def completion(
        self,
        model: str,
        content: InterleavedContent,
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
    ) -> AsyncGenerator:
        raise NotImplementedError()

    async def chat_completion(
        self,
        model: str,
        messages: list[Message],
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = ToolChoice.auto,
        tool_prompt_format: ToolPromptFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
        tool_config: ToolConfig | None = None,
    ) -> AsyncGenerator:
        if sampling_params is None:
            sampling_params = SamplingParams()
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            stream=stream,
            logprobs=logprobs,
            tool_config=tool_config,
        )

        client = OpenAI(base_url=self.config.url, api_key=self.config.api_token)
        if stream:
            return self._stream_chat_completion(request, client)
        else:
            return await self._nonstream_chat_completion(request, client)

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest, client: OpenAI
    ) -> ChatCompletionResponse:
        params = self._get_params(request)
        r = client.completions.create(**params)
        return process_chat_completion_response(r, request)

    async def _stream_chat_completion(self, request: ChatCompletionRequest, client: OpenAI) -> AsyncGenerator:
        params = self._get_params(request)

        async def _to_async_generator():
            s = client.completions.create(**params)
            for chunk in s:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_chat_completion_stream_response(stream, request):
            yield chunk

    def _get_params(self, request: ChatCompletionRequest) -> dict:
        return {
            "model": self.map_to_provider_model(request.model),
            "prompt": chat_completion_request_to_prompt(request),
            "stream": request.stream,
            **get_sampling_options(request.sampling_params),
        }

    async def embeddings(
        self,
        model: str,
        contents: list[str] | list[InterleavedContentItem],
        text_truncation: TextTruncation | None = TextTruncation.none,
        output_dimension: int | None = None,
        task_type: EmbeddingTaskType | None = None,
    ) -> EmbeddingsResponse:
        raise NotImplementedError()

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError()
