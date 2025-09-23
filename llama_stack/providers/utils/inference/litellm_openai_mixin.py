# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import litellm

from llama_stack.apis.common.content_types import (
    InterleavedContent,
    InterleavedContentItem,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChunk,
    EmbeddingsResponse,
    EmbeddingTaskType,
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
from llama_stack.core.request_headers import NeedsRequestProviderData
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.providers.utils.inference.openai_compat import (
    b64_encode_openai_embeddings_response,
    convert_message_to_openai_dict_new,
    convert_openai_chat_completion_choice,
    convert_openai_chat_completion_stream,
    convert_tooldef_to_openai_tool,
    get_sampling_options,
    prepare_openai_completion_params,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    interleaved_content_as_str,
)

logger = get_logger(name=__name__, category="providers::utils")


class LiteLLMOpenAIMixin(
    ModelRegistryHelper,
    InferenceProvider,
    NeedsRequestProviderData,
):
    # TODO: avoid exposing the litellm specific model names to the user.
    #       potential change: add a prefix param that gets added to the model name
    #                         when calling litellm.
    def __init__(
        self,
        model_entries,
        litellm_provider_name: str,
        api_key_from_config: str | None,
        provider_data_api_key_field: str,
        openai_compat_api_base: str | None = None,
        download_images: bool = False,
        json_schema_strict: bool = True,
    ):
        """
        Initialize the LiteLLMOpenAIMixin.

        :param model_entries: The model entries to register.
        :param api_key_from_config: The API key to use from the config.
        :param provider_data_api_key_field: The field in the provider data that contains the API key.
        :param litellm_provider_name: The name of the provider, used for model lookups.
        :param openai_compat_api_base: The base URL for OpenAI compatibility, or None if not using OpenAI compatibility.
        :param download_images: Whether to download images and convert to base64 for message conversion.
        :param json_schema_strict: Whether to use strict mode for JSON schema validation.
        """
        ModelRegistryHelper.__init__(self, model_entries)

        self.litellm_provider_name = litellm_provider_name
        self.api_key_from_config = api_key_from_config
        self.provider_data_api_key_field = provider_data_api_key_field
        self.api_base = openai_compat_api_base
        self.download_images = download_images
        self.json_schema_strict = json_schema_strict

        if openai_compat_api_base:
            self.is_openai_compat = True
        else:
            self.is_openai_compat = False

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    def get_litellm_model_name(self, model_id: str) -> str:
        # users may be using openai/ prefix in their model names. the openai/models.py did this by default.
        # model_id.startswith("openai/") is for backwards compatibility.
        return (
            f"{self.litellm_provider_name}/{model_id}"
            if self.is_openai_compat and not model_id.startswith(self.litellm_provider_name)
            else model_id
        )

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
    ) -> AsyncGenerator:
        raise NotImplementedError("LiteLLM does not support completion requests")

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
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionResponseStreamChunk]:
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

        params = await self._get_params(request)
        params["model"] = self.get_litellm_model_name(params["model"])

        logger.debug(f"params to litellm (openai compat): {params}")
        # see https://docs.litellm.ai/docs/completion/stream#async-completion
        response = await litellm.acompletion(**params)
        if stream:
            return self._stream_chat_completion(response)
        else:
            return convert_openai_chat_completion_choice(response.choices[0])

    async def _stream_chat_completion(
        self, response: litellm.ModelResponse
    ) -> AsyncIterator[ChatCompletionResponseStreamChunk]:
        async def _stream_generator():
            async for chunk in response:
                yield chunk

        async for chunk in convert_openai_chat_completion_stream(
            _stream_generator(), enable_incremental_tool_calls=True
        ):
            yield chunk

    def _add_additional_properties_recursive(self, schema):
        """
        Recursively add additionalProperties: False to all object schemas
        """
        if isinstance(schema, dict):
            if schema.get("type") == "object":
                schema["additionalProperties"] = False

                # Add required field with all property keys if properties exist
                if "properties" in schema and schema["properties"]:
                    schema["required"] = list(schema["properties"].keys())

            if "properties" in schema:
                for prop_schema in schema["properties"].values():
                    self._add_additional_properties_recursive(prop_schema)

            for key in ["anyOf", "allOf", "oneOf"]:
                if key in schema:
                    for sub_schema in schema[key]:
                        self._add_additional_properties_recursive(sub_schema)

            if "not" in schema:
                self._add_additional_properties_recursive(schema["not"])

            # Handle $defs/$ref
            if "$defs" in schema:
                for def_schema in schema["$defs"].values():
                    self._add_additional_properties_recursive(def_schema)

        return schema

    async def _get_params(self, request: ChatCompletionRequest) -> dict:
        input_dict = {}

        input_dict["messages"] = [
            await convert_message_to_openai_dict_new(m, download_images=self.download_images) for m in request.messages
        ]
        if fmt := request.response_format:
            if not isinstance(fmt, JsonSchemaResponseFormat):
                raise ValueError(
                    f"Unsupported response format: {type(fmt)}. Only JsonSchemaResponseFormat is supported."
                )

            fmt = fmt.json_schema
            name = fmt["title"]
            del fmt["title"]
            fmt["additionalProperties"] = False

            # Apply additionalProperties: False recursively to all objects
            fmt = self._add_additional_properties_recursive(fmt)

            input_dict["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": fmt,
                    "strict": self.json_schema_strict,
                },
            }
        if request.tools:
            input_dict["tools"] = [convert_tooldef_to_openai_tool(tool) for tool in request.tools]
            if request.tool_config.tool_choice:
                input_dict["tool_choice"] = (
                    request.tool_config.tool_choice.value
                    if isinstance(request.tool_config.tool_choice, ToolChoice)
                    else request.tool_config.tool_choice
                )

        return {
            "model": request.model,
            "api_key": self.get_api_key(),
            "api_base": self.api_base,
            **input_dict,
            "stream": request.stream,
            **get_sampling_options(request.sampling_params),
        }

    def get_api_key(self) -> str:
        provider_data = self.get_request_provider_data()
        key_field = self.provider_data_api_key_field
        if provider_data and getattr(provider_data, key_field, None):
            api_key = getattr(provider_data, key_field)
        else:
            api_key = self.api_key_from_config
        if not api_key:
            raise ValueError(
                "API key is not set. Please provide a valid API key in the "
                "provider data header, e.g. x-llamastack-provider-data: "
                f'{{"{key_field}": "<API_KEY>"}}, or in the provider config.'
            )
        return api_key

    async def embeddings(
        self,
        model_id: str,
        contents: list[str] | list[InterleavedContentItem],
        text_truncation: TextTruncation | None = TextTruncation.none,
        output_dimension: int | None = None,
        task_type: EmbeddingTaskType | None = None,
    ) -> EmbeddingsResponse:
        model = await self.model_store.get_model(model_id)

        response = litellm.embedding(
            model=self.get_litellm_model_name(model.provider_resource_id),
            input=[interleaved_content_as_str(content) for content in contents],
        )

        embeddings = [data["embedding"] for data in response["data"]]
        return EmbeddingsResponse(embeddings=embeddings)

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        model_obj = await self.model_store.get_model(model)

        # Convert input to list if it's a string
        input_list = [input] if isinstance(input, str) else input

        # Call litellm embedding function
        # litellm.drop_params = True
        response = litellm.embedding(
            model=self.get_litellm_model_name(model_obj.provider_resource_id),
            input=input_list,
            api_key=self.get_api_key(),
            api_base=self.api_base,
            dimensions=dimensions,
        )

        # Convert response to OpenAI format
        data = b64_encode_openai_embeddings_response(response.data, encoding_format)

        usage = OpenAIEmbeddingUsage(
            prompt_tokens=response["usage"]["prompt_tokens"],
            total_tokens=response["usage"]["total_tokens"],
        )

        return OpenAIEmbeddingsResponse(
            data=data,
            model=model_obj.provider_resource_id,
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
        model_obj = await self.model_store.get_model(model)
        params = await prepare_openai_completion_params(
            model=self.get_litellm_model_name(model_obj.provider_resource_id),
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
            guided_choice=guided_choice,
            prompt_logprobs=prompt_logprobs,
            api_key=self.get_api_key(),
            api_base=self.api_base,
        )
        return await litellm.atext_completion(**params)

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
        model_obj = await self.model_store.get_model(model)
        params = await prepare_openai_completion_params(
            model=self.get_litellm_model_name(model_obj.provider_resource_id),
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
            api_key=self.get_api_key(),
            api_base=self.api_base,
        )
        return await litellm.acompletion(**params)

    async def check_model_availability(self, model: str) -> bool:
        """
        Check if a specific model is available via LiteLLM for the current
        provider (self.litellm_provider_name).

        :param model: The model identifier to check.
        :return: True if the model is available dynamically, False otherwise.
        """
        if self.litellm_provider_name not in litellm.models_by_provider:
            logger.error(f"Provider {self.litellm_provider_name} is not registered in litellm.")
            return False

        return model in litellm.models_by_provider[self.litellm_provider_name]
