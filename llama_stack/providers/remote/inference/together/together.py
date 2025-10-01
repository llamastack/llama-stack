# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncGenerator

from openai import AsyncOpenAI
from together import AsyncTogether
from together.constants import BASE_URL

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Inference,
    LogProbConfig,
    Message,
    OpenAIEmbeddingsResponse,
    ResponseFormat,
    ResponseFormatType,
    SamplingParams,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.apis.inference.inference import OpenAIEmbeddingUsage
from llama_stack.apis.models import Model, ModelType
from llama_stack.core.request_headers import NeedsRequestProviderData
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.providers.utils.inference.openai_compat import (
    convert_message_to_openai_dict,
    get_sampling_options,
    process_chat_completion_response,
    process_chat_completion_stream_response,
)
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
    request_has_media,
)

from .config import TogetherImplConfig

logger = get_logger(name=__name__, category="inference::together")


class TogetherInferenceAdapter(OpenAIMixin, ModelRegistryHelper, Inference, NeedsRequestProviderData):
    embedding_model_metadata = {
        "togethercomputer/m2-bert-80M-32k-retrieval": {"embedding_dimension": 768, "context_length": 32768},
        "BAAI/bge-large-en-v1.5": {"embedding_dimension": 1024, "context_length": 512},
        "BAAI/bge-base-en-v1.5": {"embedding_dimension": 768, "context_length": 512},
        "Alibaba-NLP/gte-modernbert-base": {"embedding_dimension": 768, "context_length": 8192},
        "intfloat/multilingual-e5-large-instruct": {"embedding_dimension": 1024, "context_length": 512},
    }

    def __init__(self, config: TogetherImplConfig) -> None:
        ModelRegistryHelper.__init__(self)
        self.config = config
        self.allowed_models = config.allowed_models
        self._model_cache: dict[str, Model] = {}

    def get_api_key(self):
        return self.config.api_key.get_secret_value()

    def get_base_url(self):
        return BASE_URL

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def _get_client(self) -> AsyncTogether:
        together_api_key = None
        config_api_key = self.config.api_key.get_secret_value() if self.config.api_key else None
        if config_api_key:
            together_api_key = config_api_key
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.together_api_key:
                raise ValueError(
                    'Pass Together API Key in the header X-LlamaStack-Provider-Data as { "together_api_key": <your api key>}'
                )
            together_api_key = provider_data.together_api_key
        return AsyncTogether(api_key=together_api_key)

    def _get_openai_client(self) -> AsyncOpenAI:
        together_client = self._get_client().client
        return AsyncOpenAI(
            base_url=together_client.base_url,
            api_key=together_client.api_key,
        )

    def _build_options(
        self,
        sampling_params: SamplingParams | None,
        logprobs: LogProbConfig | None,
        fmt: ResponseFormat,
    ) -> dict:
        options = get_sampling_options(sampling_params)
        if fmt:
            if fmt.type == ResponseFormatType.json_schema.value:
                options["response_format"] = {
                    "type": "json_object",
                    "schema": fmt.json_schema,
                }
            elif fmt.type == ResponseFormatType.grammar.value:
                raise NotImplementedError("Grammar response format not supported yet")
            else:
                raise ValueError(f"Unknown response format {fmt.type}")

        if logprobs and logprobs.top_k:
            if logprobs.top_k != 1:
                raise ValueError(
                    f"Unsupported value: Together only supports logprobs top_k=1. {logprobs.top_k} was provided",
                )
            options["logprobs"] = 1

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
        client = self._get_client()
        if "messages" in params:
            r = await client.chat.completions.create(**params)
        else:
            r = await client.completions.create(**params)
        return process_chat_completion_response(r, request)

    async def _stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)
        client = self._get_client()
        if "messages" in params:
            stream = await client.chat.completions.create(**params)
        else:
            stream = await client.completions.create(**params)

        async for chunk in process_chat_completion_stream_response(stream, request):
            yield chunk

    async def _get_params(self, request: ChatCompletionRequest) -> dict:
        input_dict = {}
        media_present = request_has_media(request)
        llama_model = self.get_llama_model(request.model)
        if media_present or not llama_model:
            input_dict["messages"] = [await convert_message_to_openai_dict(m) for m in request.messages]
        else:
            input_dict["prompt"] = await chat_completion_request_to_prompt(request, llama_model)

        params = {
            "model": request.model,
            **input_dict,
            "stream": request.stream,
            **self._build_options(request.sampling_params, request.logprobs, request.response_format),
        }
        logger.debug(f"params to together: {params}")
        return params

    async def list_models(self) -> list[Model] | None:
        self._model_cache = {}
        # Together's /v1/models is not compatible with OpenAI's /v1/models. Together support ticket #13355 -> will not fix, use Together's own client
        for m in await self._get_client().models.list():
            if m.type == "embedding":
                if m.id not in self.embedding_model_metadata:
                    logger.warning(f"Unknown embedding dimension for model {m.id}, skipping.")
                    continue
                metadata = self.embedding_model_metadata[m.id]
                self._model_cache[m.id] = Model(
                    provider_id=self.__provider_id__,
                    provider_resource_id=m.id,
                    identifier=m.id,
                    model_type=ModelType.embedding,
                    metadata=metadata,
                )
            else:
                self._model_cache[m.id] = Model(
                    provider_id=self.__provider_id__,
                    provider_resource_id=m.id,
                    identifier=m.id,
                    model_type=ModelType.llm,
                )

        return self._model_cache.values()

    async def should_refresh_models(self) -> bool:
        return True

    async def check_model_availability(self, model):
        return model in self._model_cache

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        """
        Together's OpenAI-compatible embeddings endpoint is not compatible with
        the standard OpenAI embeddings endpoint.

        The endpoint -
         - not all models return usage information
         - does not support user param, returns 400 Unrecognized request arguments supplied: user
         - does not support dimensions param, returns 400 Unrecognized request arguments supplied: dimensions
        """
        # Together support ticket #13332 -> will not fix
        if user is not None:
            raise ValueError("Together's embeddings endpoint does not support user param.")
        # Together support ticket #13333 -> escalated
        if dimensions is not None:
            raise ValueError("Together's embeddings endpoint does not support dimensions param.")

        response = await self.client.embeddings.create(
            model=await self._get_provider_model_id(model),
            input=input,
            encoding_format=encoding_format,
        )

        response.model = model  # return the user the same model id they provided, avoid exposing the provider model id

        # Together support ticket #13330 -> escalated
        #  - togethercomputer/m2-bert-80M-32k-retrieval *does not* return usage information
        if not hasattr(response, "usage") or response.usage is None:
            logger.warning(
                f"Together's embedding endpoint for {model} did not return usage information, substituting -1s."
            )
            response.usage = OpenAIEmbeddingUsage(prompt_tokens=-1, total_tokens=-1)

        return response
