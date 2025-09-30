# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncGenerator
from openai import OpenAI
from llama_stack.apis.inference import *
from llama_stack.apis.inference import OpenAIEmbeddingsResponse
from llama_stack.apis.models import Model, ModelType
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.providers.utils.inference.openai_compat import (
    OpenAIChatCompletionToLlamaStackMixin,
    OpenAICompletionToLlamaStackMixin,
    get_sampling_options,
    process_chat_completion_response,
    process_chat_completion_stream_response,
    process_completion_response,
    process_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    completion_request_to_prompt,
    interleaved_content_as_str,
)
from .config import RunpodImplConfig

MODEL_ENTRIES = []


class RunpodInferenceAdapter(
    ModelRegistryHelper,
    Inference,
    OpenAIChatCompletionToLlamaStackMixin,
    OpenAICompletionToLlamaStackMixin,
):
    """
    Adapter for RunPod's OpenAI-compatible API endpoints.
    Supports VLLM for serverless endpoint self-hosted or public endpoints.
    Can work with any runpod endpoints that support OpenAI-compatible API
    """

    def __init__(self, config: RunpodImplConfig) -> None:
        ModelRegistryHelper.__init__(self, MODEL_ENTRIES)
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_model(self, model: Model) -> Model:
        """
        Register any model with the runpod provider_id.

        Pass-through registration - accepts any model string that the RunPod endpoint serves.
        No static model validation since RunPod endpoints can serve arbitrary vLLM models.

        YAML Configuration Example:
            models:
            - metadata: {}
                model_id: runpod/qwen/qwen3-8b
                model_type: llm
                provider_id: runpod
                provider_model_id: qwen/qwen3-8b
            - metadata: {}
                model_id: runpod/deepcogito/cogito-v2-preview-llama-70B
                model_type: llm
                provider_id: runpod
                provider_model_id: deepcogito/cogito-v2-preview-llama-70B

        The provider strips 'runpod/' prefix before API calls:
            "runpod/qwen/qwen3-8b" -> "qwen/qwen3-8b"
        """
        if model.provider_id == "runpod":
            logger.info(
                f"Registering model: {model.identifier} -> {model.provider_resource_id}"
            )
            return model
        return await super().register_model(model)

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
    ) -> AsyncGenerator:
        if sampling_params is None:
            sampling_params = SamplingParams()

        request = CompletionRequest(
            model=model_id,
            content=content,
            sampling_params=sampling_params,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )

        client = OpenAI(base_url=self.config.url, api_key=self.config.api_token)

        if stream:
            return self._stream_completion(request, client)
        else:
            return await self._nonstream_completion(request, client)

    async def chat_completion(
        self,
        model_id: str,
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
        """Process chat completion requests using RunPod's OpenAI-compatible API."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        request = ChatCompletionRequest(
            model=model_id,
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
        params = await self._get_chat_params(request)
        r = client.chat.completions.create(**params)
        return process_chat_completion_response(r, request)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest, client: OpenAI
    ) -> AsyncGenerator:
        params = await self._get_chat_params(request)

        async def _to_async_generator():
            s = client.chat.completions.create(**params)
            for chunk in s:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_chat_completion_stream_response(stream, request):
            yield chunk

    async def _get_chat_params(self, request: ChatCompletionRequest) -> dict:
        """Convert Llama Stack request to RunPod API parameters."""
        messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        # Resolve model_id to provider_resource_id
        model_obj = await self.model_store.get_model(request.model)
        model = model_obj.provider_resource_id or request.model

        if model.startswith("runpod/"):
            model = model.replace("runpod/", "", 1)

        params = {
            "model": model,
            "messages": messages,
            "stream": request.stream,
            **get_sampling_options(request.sampling_params),
        }

        if request.stream:
            params["stream_options"] = {"include_usage": True}

        return params

    async def _nonstream_completion(
        self, request: CompletionRequest, client: OpenAI
    ) -> CompletionResponse:
        params = await self._get_completion_params(request)
        r = client.completions.create(**params)
        return process_completion_response(r)

    async def _stream_completion(
        self, request: CompletionRequest, client: OpenAI
    ) -> AsyncGenerator:
        params = await self._get_completion_params(request)

        async def _to_async_generator():
            s = client.completions.create(**params)
            for chunk in s:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_completion_stream_response(stream):
            yield chunk

    async def _get_completion_params(self, request: CompletionRequest) -> dict:
        # Resolve model_id to provider_resource_id
        model_obj = await self.model_store.get_model(request.model)
        model = model_obj.provider_resource_id or request.model

        if model.startswith("runpod/"):
            model = model.replace("runpod/", "", 1)

        params = {
            "model": model,
            "prompt": completion_request_to_prompt(request),
            "stream": request.stream,
            **get_sampling_options(request.sampling_params),
        }

        if request.stream:
            params["stream_options"] = {"include_usage": True}

        return params

    async def embeddings(
        self,
        model_id: str,
        contents: list[str] | list[InterleavedContentItem],
        text_truncation: TextTruncation | None = TextTruncation.none,
        output_dimension: int | None = None,
        task_type: EmbeddingTaskType | None = None,
    ) -> EmbeddingsResponse:
        # Resolve model_id to provider_resource_id
        model_obj = await self.model_store.get_model(model_id)
        model = model_obj.provider_resource_id or model_id

        if model.startswith("runpod/"):
            model = model.replace("runpod/", "", 1)

        client = OpenAI(base_url=self.config.url, api_key=self.config.api_token)

        kwargs = {}
        if output_dimension:
            kwargs["dimensions"] = output_dimension

        response = client.embeddings.create(
            model=model,
            input=[interleaved_content_as_str(content) for content in contents],
            **kwargs,
        )

        embeddings = [data.embedding for data in response.data]
        return EmbeddingsResponse(embeddings=embeddings)

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        # Resolve model_id to provider_resource_id
        model_obj = await self.model_store.get_model(model)
        model_stripped = model_obj.provider_resource_id or model

        if model_stripped.startswith("runpod/"):
            model_stripped = model_stripped.replace("runpod/", "", 1)

        client = OpenAI(base_url=self.config.url, api_key=self.config.api_token)

        response = client.embeddings.create(
            model=model_stripped,
            input=input,
            encoding_format=encoding_format,
            dimensions=dimensions,
            user=user,
        )

        return response