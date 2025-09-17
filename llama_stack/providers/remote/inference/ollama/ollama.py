# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import asyncio
import base64
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ollama import AsyncClient  # type: ignore[attr-defined]
from openai import AsyncOpenAI

from llama_stack.apis.common.content_types import (
    ImageContentItem,
    InterleavedContent,
    InterleavedContentItem,
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
    EmbeddingsResponse,
    EmbeddingTaskType,
    GrammarResponseFormat,
    InferenceProvider,
    JsonSchemaResponseFormat,
    LogProbConfig,
    Message,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAICompletion,
    OpenAIEmbeddingsResponse,
    OpenAIEmbeddingUsage,
    OpenAIMessageParam,
    OpenAIResponseFormatParam,
    ResponseFormat,
    SamplingParams,
    TextTruncation,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.apis.models import Model, ModelType
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import (
    HealthResponse,
    HealthStatus,
    ModelsProtocolPrivate,
)
from llama_stack.providers.remote.inference.ollama.config import OllamaImplConfig
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
)
from llama_stack.providers.utils.inference.openai_compat import (
    OpenAICompatCompletionChoice,
    OpenAICompatCompletionResponse,
    b64_encode_openai_embeddings_response,
    get_sampling_options,
    prepare_openai_completion_params,
    prepare_openai_embeddings_params,
    process_chat_completion_response,
    process_chat_completion_stream_response,
    process_completion_response,
    process_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
    completion_request_to_prompt,
    content_has_media,
    convert_image_content_to_url,
    interleaved_content_as_str,
    localize_image_content,
    request_has_media,
)

from .models import MODEL_ENTRIES

logger = get_logger(name=__name__, category="inference::ollama")


class OllamaInferenceAdapter(
    InferenceProvider,
    ModelsProtocolPrivate,
):
    # automatically set by the resolver when instantiating the provider
    __provider_id__: str

    def __init__(self, config: OllamaImplConfig) -> None:
        self.register_helper = ModelRegistryHelper(MODEL_ENTRIES)
        self.config = config
        self._clients: dict[asyncio.AbstractEventLoop, AsyncClient] = {}
        self._openai_client = None

    @property
    def client(self) -> AsyncClient:
        # ollama client attaches itself to the current event loop (sadly?)
        loop = asyncio.get_running_loop()
        if loop not in self._clients:
            self._clients[loop] = AsyncClient(host=self.config.url)
        return self._clients[loop]

    @property
    def openai_client(self) -> AsyncOpenAI:
        if self._openai_client is None:
            url = self.config.url.rstrip("/")
            self._openai_client = AsyncOpenAI(base_url=f"{url}/v1", api_key="ollama")
        return self._openai_client

    async def initialize(self) -> None:
        logger.info(f"checking connectivity to Ollama at `{self.config.url}`...")
        r = await self.health()
        if r["status"] == HealthStatus.ERROR:
            logger.warning(
                f"Ollama Server is not running (message: {r['message']}). Make sure to start it using `ollama serve` in a separate terminal"
            )

    async def should_refresh_models(self) -> bool:
        return self.config.refresh_models

    async def list_models(self) -> list[Model] | None:
        provider_id = self.__provider_id__
        response = await self.client.list()

        # always add the two embedding models which can be pulled on demand
        models = [
            Model(
                identifier="all-minilm:l6-v2",
                provider_resource_id="all-minilm:l6-v2",
                provider_id=provider_id,
                metadata={
                    "embedding_dimension": 384,
                    "context_length": 512,
                },
                model_type=ModelType.embedding,
            ),
            # add all-minilm alias
            Model(
                identifier="all-minilm",
                provider_resource_id="all-minilm:l6-v2",
                provider_id=provider_id,
                metadata={
                    "embedding_dimension": 384,
                    "context_length": 512,
                },
                model_type=ModelType.embedding,
            ),
            Model(
                identifier="nomic-embed-text",
                provider_resource_id="nomic-embed-text:latest",
                provider_id=provider_id,
                metadata={
                    "embedding_dimension": 768,
                    "context_length": 8192,
                },
                model_type=ModelType.embedding,
            ),
        ]
        for m in response.models:
            # kill embedding models since we don't know dimensions for them
            if "bert" in m.details.family:
                continue
            models.append(
                Model(
                    identifier=m.model,
                    provider_resource_id=m.model,
                    provider_id=provider_id,
                    metadata={},
                    model_type=ModelType.llm,
                )
            )
        return models

    async def health(self) -> HealthResponse:
        """
        Performs a health check by verifying connectivity to the Ollama server.
        This method is used by initialize() and the Provider API to verify that the service is running
        correctly.
        Returns:
            HealthResponse: A dictionary containing the health status.
        """
        try:
            await self.client.ps()
            return HealthResponse(status=HealthStatus.OK)
        except Exception as e:
            return HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}")

    async def shutdown(self) -> None:
        self._clients.clear()

    async def unregister_model(self, model_id: str) -> None:
        pass

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
            s = await self.client.generate(**params)
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
        r = await self.client.generate(**params)

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
        llama_model = self.register_helper.get_llama_model(request.model)
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
            r = await self.client.chat(**params)
        else:
            r = await self.client.generate(**params)

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
                s = await self.client.chat(**params)
            else:
                s = await self.client.generate(**params)
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

    async def embeddings(
        self,
        model_id: str,
        contents: list[str] | list[InterleavedContentItem],
        text_truncation: TextTruncation | None = TextTruncation.none,
        output_dimension: int | None = None,
        task_type: EmbeddingTaskType | None = None,
    ) -> EmbeddingsResponse:
        model = await self._get_model(model_id)

        assert all(not content_has_media(content) for content in contents), (
            "Ollama does not support media for embeddings"
        )
        response = await self.client.embed(
            model=model.provider_resource_id,
            input=[interleaved_content_as_str(content) for content in contents],
        )
        embeddings = response["embeddings"]

        return EmbeddingsResponse(embeddings=embeddings)

    async def register_model(self, model: Model) -> Model:
        try:
            model = await self.register_helper.register_model(model)
        except ValueError:
            pass  # Ignore statically unknown model, will check live listing

        if model.model_type == ModelType.embedding:
            response = await self.client.list()
            if model.provider_resource_id not in [m.model for m in response.models]:
                await self.client.pull(model.provider_resource_id)

        # we use list() here instead of ps() -
        #  - ps() only lists running models, not available models
        #  - models not currently running are run by the ollama server as needed
        response = await self.client.list()
        available_models = [m.model for m in response.models]

        provider_resource_id = model.provider_resource_id
        assert provider_resource_id is not None  # mypy
        if provider_resource_id not in available_models:
            available_models_latest = [m.model.split(":latest")[0] for m in response.models]
            if provider_resource_id in available_models_latest:
                logger.warning(
                    f"Imprecise provider resource id was used but 'latest' is available in Ollama - using '{model.provider_resource_id}:latest'"
                )
                return model
            raise UnsupportedModelError(provider_resource_id, available_models)

        # mutating this should be considered an anti-pattern
        model.provider_resource_id = provider_resource_id

        return model

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        model_obj = await self._get_model(model)
        if model_obj.provider_resource_id is None:
            raise ValueError(f"Model {model} has no provider_resource_id set")

        # Note, at the moment Ollama does not support encoding_format, dimensions, and user parameters
        params = prepare_openai_embeddings_params(
            model=model_obj.provider_resource_id,
            input=input,
            encoding_format=encoding_format,
            dimensions=dimensions,
            user=user,
        )

        response = await self.openai_client.embeddings.create(**params)
        data = b64_encode_openai_embeddings_response(response.data, encoding_format)

        usage = OpenAIEmbeddingUsage(
            prompt_tokens=response.usage.prompt_tokens,
            total_tokens=response.usage.total_tokens,
        )
        # TODO: Investigate why model_obj.identifier is used instead of response.model
        return OpenAIEmbeddingsResponse(
            data=data,
            model=model_obj.identifier,
            usage=usage,
        )

    async def openai_completion(
        self,
        model: str,
        prompt: str | list[str] | list[int] | list[list[int]],
        best_of: int | None = None,
        echo: bool | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream: bool | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        user: str | None = None,
        guided_choice: list[str] | None = None,
        prompt_logprobs: int | None = None,
        suffix: str | None = None,
    ) -> OpenAICompletion:
        if not isinstance(prompt, str):
            raise ValueError("Ollama does not support non-string prompts for completion")

        model_obj = await self._get_model(model)
        params = await prepare_openai_completion_params(
            model=model_obj.provider_resource_id,
            prompt=prompt,
            best_of=best_of,
            echo=echo,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=stop,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
            top_p=top_p,
            user=user,
            suffix=suffix,
        )
        return await self.openai_client.completions.create(**params)  # type: ignore

    async def openai_chat_completion(
        self,
        model: str,
        messages: list[OpenAIMessageParam],
        frequency_penalty: float | None = None,
        function_call: str | dict[str, Any] | None = None,
        functions: list[dict[str, Any]] | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        presence_penalty: float | None = None,
        response_format: OpenAIResponseFormatParam | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream: bool | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        user: str | None = None,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        model_obj = await self._get_model(model)

        # Ollama does not support image urls, so we need to download the image and convert it to base64
        async def _convert_message(m: OpenAIMessageParam) -> OpenAIMessageParam:
            if isinstance(m.content, list):
                for c in m.content:
                    if c.type == "image_url" and c.image_url and c.image_url.url:
                        localize_result = await localize_image_content(c.image_url.url)
                        if localize_result is None:
                            raise ValueError(f"Failed to localize image content from {c.image_url.url}")

                        content, format = localize_result
                        c.image_url.url = f"data:image/{format};base64,{base64.b64encode(content).decode('utf-8')}"
            return m

        messages = [await _convert_message(m) for m in messages]
        params = await prepare_openai_completion_params(
            model=model_obj.provider_resource_id,
            messages=messages,
            frequency_penalty=frequency_penalty,
            function_call=function_call,
            functions=functions,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            n=n,
            parallel_tool_calls=parallel_tool_calls,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
            tool_choice=tool_choice,
            tools=tools,
            top_logprobs=top_logprobs,
            top_p=top_p,
            user=user,
        )
        response = await self.openai_client.chat.completions.create(**params)
        return await self._adjust_ollama_chat_completion_response_ids(response)

    async def _adjust_ollama_chat_completion_response_ids(
        self,
        response: OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk],
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        id = f"chatcmpl-{uuid.uuid4()}"
        if isinstance(response, AsyncIterator):

            async def stream_with_chunk_ids() -> AsyncIterator[OpenAIChatCompletionChunk]:
                async for chunk in response:
                    chunk.id = id
                    yield chunk

            return stream_with_chunk_ids()
        else:
            response.id = id
            return response


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
