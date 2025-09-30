# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from ollama import AsyncClient as AsyncOllamaClient

from llama_stack.apis.common.content_types import (
    ImageContentItem,
    InterleavedContent,
    TextContentItem,
)
from llama_stack.apis.common.errors import UnsupportedModelError
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChunk,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseStreamChunk,
    GrammarResponseFormat,
    InferenceProvider,
    JsonSchemaResponseFormat,
    LogProbConfig,
    Message,
    ResponseFormat,
    SamplingParams,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.apis.models import Model
from llama_stack.log import get_logger
from llama_stack.models.llama.sku_types import CoreModelId
from llama_stack.providers.datatypes import (
    HealthResponse,
    HealthStatus,
    ModelsProtocolPrivate,
)
from llama_stack.providers.remote.inference.ollama.config import OllamaImplConfig
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
    build_hf_repo_model_entry,
)
from llama_stack.providers.utils.inference.openai_compat import (
    OpenAICompatCompletionChoice,
    OpenAICompatCompletionResponse,
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
    convert_image_content_to_url,
    request_has_media,
)

logger = get_logger(name=__name__, category="inference::ollama")


class OllamaInferenceAdapter(
    OpenAIMixin,
    ModelRegistryHelper,
    InferenceProvider,
    ModelsProtocolPrivate,
):
    # automatically set by the resolver when instantiating the provider
    __provider_id__: str

    embedding_model_metadata = {
        "all-minilm:l6-v2": {
            "embedding_dimension": 384,
            "context_length": 512,
        },
        "nomic-embed-text:latest": {
            "embedding_dimension": 768,
            "context_length": 8192,
        },
        "nomic-embed-text:v1.5": {
            "embedding_dimension": 768,
            "context_length": 8192,
        },
        "nomic-embed-text:137m-v1.5-fp16": {
            "embedding_dimension": 768,
            "context_length": 8192,
        },
    }

    def __init__(self, config: OllamaImplConfig) -> None:
        # TODO: remove ModelRegistryHelper.__init__ when completion and
        #       chat_completion are. this exists to satisfy the input /
        #       output processing for llama models. specifically,
        #       tool_calling is handled by raw template processing,
        #       instead of using the /api/chat endpoint w/ tools=...
        ModelRegistryHelper.__init__(
            self,
            model_entries=[
                build_hf_repo_model_entry(
                    "llama3.2:3b-instruct-fp16",
                    CoreModelId.llama3_2_3b_instruct.value,
                ),
                build_hf_repo_model_entry(
                    "llama-guard3:1b",
                    CoreModelId.llama_guard_3_1b.value,
                ),
            ],
        )
        self.config = config
        # Ollama does not support image urls, so we need to download the image and convert it to base64
        self.download_images = True
        self._clients: dict[asyncio.AbstractEventLoop, AsyncOllamaClient] = {}

    @property
    def ollama_client(self) -> AsyncOllamaClient:
        # ollama client attaches itself to the current event loop (sadly?)
        loop = asyncio.get_running_loop()
        if loop not in self._clients:
            self._clients[loop] = AsyncOllamaClient(host=self.config.url)
        return self._clients[loop]

    def get_api_key(self):
        return "NO_KEY"

    def get_base_url(self):
        return self.config.url.rstrip("/") + "/v1"

    async def initialize(self) -> None:
        logger.info(f"checking connectivity to Ollama at `{self.config.url}`...")
        r = await self.health()
        if r["status"] == HealthStatus.ERROR:
            logger.warning(
                f"Ollama Server is not running (message: {r['message']}). Make sure to start it using `ollama serve` in a separate terminal"
            )

    async def should_refresh_models(self) -> bool:
        return self.config.refresh_models

    async def health(self) -> HealthResponse:
        """
        Performs a health check by verifying connectivity to the Ollama server.
        This method is used by initialize() and the Provider API to verify that the service is running
        correctly.
        Returns:
            HealthResponse: A dictionary containing the health status.
        """
        try:
            await self.ollama_client.ps()
            return HealthResponse(status=HealthStatus.OK)
        except Exception as e:
            return HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}")

    async def shutdown(self) -> None:
        self._clients.clear()

    async def _get_model(self, model_id: str) -> Model:
        if not self.model_store:
            raise ValueError("Model store not set")
        return await self.model_store.get_model(model_id)

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
    ) -> CompletionResponse | AsyncGenerator[CompletionResponseStreamChunk, None]:
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self._get_model(model_id)
        if model.provider_resource_id is None:
            raise ValueError(f"Model {model_id} has no provider_resource_id set")
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

    async def _stream_completion(
        self, request: CompletionRequest
    ) -> AsyncGenerator[CompletionResponseStreamChunk, None]:
        params = await self._get_params(request)

        async def _generate_and_convert_to_openai_compat():
            s = await self.ollama_client.generate(**params)
            async for chunk in s:
                choice = OpenAICompatCompletionChoice(
                    finish_reason=chunk["done_reason"] if chunk["done"] else None,
                    text=chunk["response"],
                )
                yield OpenAICompatCompletionResponse(
                    choices=[choice],
                )

        stream = _generate_and_convert_to_openai_compat()
        async for chunk in process_completion_stream_response(stream):
            yield chunk

    async def _nonstream_completion(self, request: CompletionRequest) -> CompletionResponse:
        params = await self._get_params(request)
        r = await self.ollama_client.generate(**params)

        choice = OpenAICompatCompletionChoice(
            finish_reason=r["done_reason"] if r["done"] else None,
            text=r["response"],
        )
        response = OpenAICompatCompletionResponse(
            choices=[choice],
        )

        return process_completion_response(response)

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
    ) -> ChatCompletionResponse | AsyncGenerator[ChatCompletionResponseStreamChunk, None]:
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self._get_model(model_id)
        if model.provider_resource_id is None:
            raise ValueError(f"Model {model_id} has no provider_resource_id set")
        request = ChatCompletionRequest(
            model=model.provider_resource_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            stream=stream,
            logprobs=logprobs,
            response_format=response_format,
            tool_config=tool_config,
        )
        if stream:
            return self._stream_chat_completion(request)
        else:
            return await self._nonstream_chat_completion(request)

    async def _get_params(self, request: ChatCompletionRequest | CompletionRequest) -> dict:
        sampling_options = get_sampling_options(request.sampling_params)
        # This is needed since the Ollama API expects num_predict to be set
        # for early truncation instead of max_tokens.
        if sampling_options.get("max_tokens") is not None:
            sampling_options["num_predict"] = sampling_options["max_tokens"]

        input_dict: dict[str, Any] = {}
        media_present = request_has_media(request)
        llama_model = self.get_llama_model(request.model)
        if isinstance(request, ChatCompletionRequest):
            if media_present or not llama_model:
                contents = [await convert_message_to_openai_dict_for_ollama(m) for m in request.messages]
                # flatten the list of lists
                input_dict["messages"] = [item for sublist in contents for item in sublist]
            else:
                input_dict["raw"] = True
                input_dict["prompt"] = await chat_completion_request_to_prompt(
                    request,
                    llama_model,
                )
        else:
            assert not media_present, "Ollama does not support media for Completion requests"
            input_dict["prompt"] = await completion_request_to_prompt(request)
            input_dict["raw"] = True

        if fmt := request.response_format:
            if isinstance(fmt, JsonSchemaResponseFormat):
                input_dict["format"] = fmt.json_schema
            elif isinstance(fmt, GrammarResponseFormat):
                raise NotImplementedError("Grammar response format is not supported")
            else:
                raise ValueError(f"Unknown response format type: {fmt.type}")

        params = {
            "model": request.model,
            **input_dict,
            "options": sampling_options,
            "stream": request.stream,
        }
        logger.debug(f"params to ollama: {params}")

        return params

    async def _nonstream_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        params = await self._get_params(request)
        if "messages" in params:
            r = await self.ollama_client.chat(**params)
        else:
            r = await self.ollama_client.generate(**params)

        if "message" in r:
            choice = OpenAICompatCompletionChoice(
                finish_reason=r["done_reason"] if r["done"] else None,
                text=r["message"]["content"],
            )
        else:
            choice = OpenAICompatCompletionChoice(
                finish_reason=r["done_reason"] if r["done"] else None,
                text=r["response"],
            )
        response = OpenAICompatCompletionResponse(
            choices=[choice],
        )
        return process_chat_completion_response(response, request)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionResponseStreamChunk, None]:
        params = await self._get_params(request)

        async def _generate_and_convert_to_openai_compat():
            if "messages" in params:
                s = await self.ollama_client.chat(**params)
            else:
                s = await self.ollama_client.generate(**params)
            async for chunk in s:
                if "message" in chunk:
                    choice = OpenAICompatCompletionChoice(
                        finish_reason=chunk["done_reason"] if chunk["done"] else None,
                        text=chunk["message"]["content"],
                    )
                else:
                    choice = OpenAICompatCompletionChoice(
                        finish_reason=chunk["done_reason"] if chunk["done"] else None,
                        text=chunk["response"],
                    )
                yield OpenAICompatCompletionResponse(
                    choices=[choice],
                )

        stream = _generate_and_convert_to_openai_compat()
        async for chunk in process_chat_completion_stream_response(stream, request):
            yield chunk

    async def register_model(self, model: Model) -> Model:
        if await self.check_model_availability(model.provider_model_id):
            return model
        elif await self.check_model_availability(f"{model.provider_model_id}:latest"):
            model.provider_resource_id = f"{model.provider_model_id}:latest"
            logger.warning(
                f"Imprecise provider resource id was used but 'latest' is available in Ollama - using '{model.provider_model_id}'"
            )
            return model

        raise UnsupportedModelError(model.provider_model_id, list(self._model_cache.keys()))


async def convert_message_to_openai_dict_for_ollama(message: Message) -> list[dict]:
    async def _convert_content(content) -> dict:
        if isinstance(content, ImageContentItem):
            return {
                "role": message.role,
                "images": [await convert_image_content_to_url(content, download=True, include_format=False)],
            }
        else:
            text = content.text if isinstance(content, TextContentItem) else content
            assert isinstance(text, str)
            return {
                "role": message.role,
                "content": text,
            }

    if isinstance(message.content, list):
        return [await _convert_content(c) for c in message.content]
    else:
        return [await _convert_content(message.content)]
