# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncGenerator

from fireworks.client import Fireworks

from llama_stack.apis.common.content_types import (
    InterleavedContent,
    InterleavedContentItem,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    EmbeddingsResponse,
    EmbeddingTaskType,
    Inference,
    LogProbConfig,
    Message,
    ResponseFormat,
    ResponseFormatType,
    SamplingParams,
    TextTruncation,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.core.request_headers import NeedsRequestProviderData
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
)
from llama_stack.providers.utils.inference.openai_compat import (
    convert_message_to_openai_dict,
    get_sampling_options,
    process_chat_completion_response,
    process_chat_completion_stream_response,
    process_completion_response,
    process_completion_stream_response,
)
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
    completion_request_to_prompt,
    content_has_media,
    interleaved_content_as_str,
    request_has_media,
)

from .config import FireworksImplConfig
from .models import MODEL_ENTRIES

logger = get_logger(name=__name__, category="inference::fireworks")


class FireworksInferenceAdapter(OpenAIMixin, ModelRegistryHelper, Inference, NeedsRequestProviderData):
    def __init__(self, config: FireworksImplConfig) -> None:
        ModelRegistryHelper.__init__(self, MODEL_ENTRIES, config.allowed_models)
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def get_api_key(self) -> str:
        config_api_key = self.config.api_key.get_secret_value() if self.config.api_key else None
        if config_api_key:
            return config_api_key
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.fireworks_api_key:
                raise ValueError(
                    'Pass Fireworks API Key in the header X-LlamaStack-Provider-Data as { "fireworks_api_key": <your api key>}'
                )
            return provider_data.fireworks_api_key

    def get_base_url(self) -> str:
        return "https://api.fireworks.ai/inference/v1"

    def _get_client(self) -> Fireworks:
        fireworks_api_key = self.get_api_key()
        return Fireworks(api_key=fireworks_api_key)

    def _preprocess_prompt_for_fireworks(self, prompt: str) -> str:
        """Remove BOS token as Fireworks automatically prepends it"""
        if prompt.startswith("<|begin_of_text|>"):
            return prompt[len("<|begin_of_text|>") :]
        return prompt

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
        model = await self.model_store.get_model(model_id)
        request = CompletionRequest(
            model=model.provider_resource_id,
            content=content,
            sampling_params=sampling_params,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )
        if stream:
            return self._stream_completion(request)
        else:
            return await self._nonstream_completion(request)

    async def _nonstream_completion(self, request: CompletionRequest) -> CompletionResponse:
        params = await self._get_params(request)
        r = await self._get_client().completion.acreate(**params)
        return process_completion_response(r)

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)

        # Wrapper for async generator similar
        async def _to_async_generator():
            stream = self._get_client().completion.create(**params)
            for chunk in stream:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_completion_stream_response(stream):
            yield chunk

    def _build_options(
        self,
        sampling_params: SamplingParams | None,
        fmt: ResponseFormat,
        logprobs: LogProbConfig | None,
    ) -> dict:
        options = get_sampling_options(sampling_params)
        options.setdefault("max_tokens", 512)

        if fmt:
            if fmt.type == ResponseFormatType.json_schema.value:
                options["response_format"] = {
                    "type": "json_object",
                    "schema": fmt.json_schema,
                }
            elif fmt.type == ResponseFormatType.grammar.value:
                options["response_format"] = {
                    "type": "grammar",
                    "grammar": fmt.bnf,
                }
            else:
                raise ValueError(f"Unknown response format {fmt.type}")

        if logprobs and logprobs.top_k:
            options["logprobs"] = logprobs.top_k
            if options["logprobs"] <= 0 or options["logprobs"] >= 5:
                raise ValueError("Required range: 0 < top_k < 5")

        return options

    async def chat_completion(
        self,
        model_id: str,
        messages: list[Message],
        sampling_params: SamplingParams | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = ToolChoice.auto,
        tool_prompt_format: ToolPromptFormat | None = None,
        response_format: ResponseFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
        tool_config: ToolConfig | None = None,
    ) -> AsyncGenerator:
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self.model_store.get_model(model_id)
        request = ChatCompletionRequest(
            model=model.provider_resource_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
            tool_config=tool_config,
        )

        if stream:
            return self._stream_chat_completion(request)
        else:
            return await self._nonstream_chat_completion(request)

    async def _nonstream_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        params = await self._get_params(request)
        if "messages" in params:
            r = await self._get_client().chat.completions.acreate(**params)
        else:
            r = await self._get_client().completion.acreate(**params)
        return process_chat_completion_response(r, request)

    async def _stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)

        async def _to_async_generator():
            if "messages" in params:
                stream = self._get_client().chat.completions.acreate(**params)
            else:
                stream = self._get_client().completion.acreate(**params)
            async for chunk in stream:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_chat_completion_stream_response(stream, request):
            yield chunk

    async def _get_params(self, request: ChatCompletionRequest | CompletionRequest) -> dict:
        input_dict = {}
        media_present = request_has_media(request)

        llama_model = self.get_llama_model(request.model)
        if isinstance(request, ChatCompletionRequest):
            # TODO: tools are never added to the request, so we need to add them here
            if media_present or not llama_model:
                input_dict["messages"] = [
                    await convert_message_to_openai_dict(m, download=True) for m in request.messages
                ]
            else:
                input_dict["prompt"] = await chat_completion_request_to_prompt(request, llama_model)
        else:
            assert not media_present, "Fireworks does not support media for Completion requests"
            input_dict["prompt"] = await completion_request_to_prompt(request)

        # Fireworks always prepends with BOS
        if "prompt" in input_dict:
            if input_dict["prompt"].startswith("<|begin_of_text|>"):
                input_dict["prompt"] = input_dict["prompt"][len("<|begin_of_text|>") :]

        params = {
            "model": request.model,
            **input_dict,
            "stream": bool(request.stream),
            **self._build_options(request.sampling_params, request.response_format, request.logprobs),
        }
        logger.debug(f"params to fireworks: {params}")

        return params

    async def embeddings(
        self,
        model_id: str,
        contents: list[str] | list[InterleavedContentItem],
        text_truncation: TextTruncation | None = TextTruncation.none,
        output_dimension: int | None = None,
        task_type: EmbeddingTaskType | None = None,
    ) -> EmbeddingsResponse:
        model = await self.model_store.get_model(model_id)

        kwargs = {}
        if model.metadata.get("embedding_dimension"):
            kwargs["dimensions"] = model.metadata.get("embedding_dimension")
        assert all(not content_has_media(content) for content in contents), (
            "Fireworks does not support media for embeddings"
        )
        response = self._get_client().embeddings.create(
            model=model.provider_resource_id,
            input=[interleaved_content_as_str(content) for content in contents],
            **kwargs,
        )

        embeddings = [data.embedding for data in response.data]
        return EmbeddingsResponse(embeddings=embeddings)
