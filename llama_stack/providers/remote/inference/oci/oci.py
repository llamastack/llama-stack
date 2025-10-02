# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import time
from collections.abc import AsyncGenerator, AsyncIterator

from llama_stack.log import get_logger

logger = get_logger(__name__)

import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    ChatDetails,
    ChatResult,
    DedicatedServingMode,
    GenericChatRequest,
    OnDemandServingMode,
    SystemMessage,
    TextContent,
    UserMessage,
)

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChunk,
    Inference,
    LogProbConfig,
    Message,
    ResponseFormat,
    SamplingParams,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.apis.inference.inference import (
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionUsage,
    OpenAIChoice,
    OpenAIChoiceDelta,
    OpenAIChunkChoice,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.openai_compat import (
    OpenAICompatCompletionChoice,
    OpenAICompatCompletionResponse,
    prepare_openai_completion_params,
    process_chat_completion_response,
    process_chat_completion_stream_response,
)

from .config import OCIConfig
from .models import OCIModelRegistryHelper

logger = get_logger(name=__name__, category="inference::oci")

OCI_AUTH_TYPE_INSTANCE_PRINCIPAL = "instance_principal"
OCI_AUTH_TYPE_CONFIG_FILE = "config_file"
VALID_OCI_AUTH_TYPES = [OCI_AUTH_TYPE_INSTANCE_PRINCIPAL, OCI_AUTH_TYPE_CONFIG_FILE]

OCI_SERVING_MODE_ON_DEMAND = "ON_DEMAND"
OCI_SERVING_MODE_DEDICATED = "DEDICATED"
VALID_OCI_SERVING_MODES = [OCI_SERVING_MODE_ON_DEMAND, OCI_SERVING_MODE_DEDICATED]


class OCIInferenceAdapter(Inference, OCIModelRegistryHelper):
    def __init__(self, config: OCIConfig) -> None:
        self.config = config
        self._client: GenerativeAiInferenceClient | None = None

        if self.config.oci_auth_type not in VALID_OCI_AUTH_TYPES:
            raise ValueError(
                f"Invalid OCI authentication type: {self.config.oci_auth_type}."
                f"Valid types are one of: {VALID_OCI_AUTH_TYPES}"
            )

        if not self.config.oci_compartment_id:
            raise ValueError("OCI_COMPARTMENT_OCID a required parameter. Either set in env variable.")

        if self.config.oci_serving_mode not in VALID_OCI_SERVING_MODES:
            raise ValueError(
                f"Invalid OCI serving mode: {self.config.oci_serving_mode}."
                f"Valid modes are one of: {VALID_OCI_SERVING_MODES}"
            )

        # Initialize with OCI-specific model registry helper after validation

        OCIModelRegistryHelper.__init__(
            self,
            compartment_id=config.oci_compartment_id or "",
            oci_config=self._get_oci_config(),
            oci_signer=self._get_oci_signer(),
        )

    @property
    def client(self) -> GenerativeAiInferenceClient:
        if self._client is None:
            self._client = self._get_client()
        return self._client

    def _get_oci_config(self) -> dict:
        if self.config.oci_auth_type == OCI_AUTH_TYPE_INSTANCE_PRINCIPAL:
            config = {"region": self.config.oci_region}
        elif self.config.oci_auth_type == OCI_AUTH_TYPE_CONFIG_FILE:
            config = oci.config.from_file(self.config.oci_config_file_path, self.config.oci_config_profile)
            if not config.get("region"):
                raise ValueError(
                    "Region not specified in config. Please specify in config or with OCI_REGION env variable."
                )

        return config

    def _get_oci_signer(self) -> oci.auth.signers.InstancePrincipalsSecurityTokenSigner | None:
        if self.config.oci_auth_type == OCI_AUTH_TYPE_INSTANCE_PRINCIPAL:
            return oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        return None

    def _get_client(self) -> GenerativeAiInferenceClient:
        if self._client is None:
            if self._get_oci_signer() is None:
                return GenerativeAiInferenceClient(config=self._get_oci_config())
            else:
                return GenerativeAiInferenceClient(
                    config=self._get_oci_config(),
                    signer=self._get_oci_signer(),
                )
        return self._client

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def _build_chat_details(self, request: ChatCompletionRequest) -> ChatDetails:
        messages = []
        system_messages = []
        user_messages = []

        for msg in request.messages:
            if msg.role == "system":
                system_messages.append(SystemMessage(name="System", content=[TextContent(text=msg.content)]))
            elif msg.role == "user":
                user_messages.append(UserMessage(name="User", content=[TextContent(text=msg.content)]))

        messages.extend(system_messages)
        messages.extend(user_messages)

        # Create chat request
        sampling_params: SamplingParams | None = request.sampling_params if request.sampling_params else None
        chat_request = GenericChatRequest(
            api_format="GENERIC",
            messages=messages,
            is_stream=request.stream,
            num_generations=1,
            seed=42,
            is_echo=False,
            top_k=-1,
            top_p=0.95,
            temperature=0.7,
            frequency_penalty=0,
            presence_penalty=sampling_params.repetition_penalty if sampling_params else 0,
            max_tokens=sampling_params.max_tokens if sampling_params else 512,
            stop=sampling_params.stop if sampling_params else None,
        )

        model_id = self.get_provider_model_id(request.model)
        if self.config.oci_serving_mode == OCI_SERVING_MODE_ON_DEMAND:
            serving_mode = OnDemandServingMode(serving_type="ON_DEMAND", model_id=model_id)
        elif self.config.oci_serving_mode == OCI_SERVING_MODE_DEDICATED:
            serving_mode = DedicatedServingMode(serving_type="DEDICATED", model_id=model_id)

        chat_details = ChatDetails(
            compartment_id=self.config.oci_compartment_id, serving_mode=serving_mode, chat_request=chat_request
        )
        return chat_details

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
        request = ChatCompletionRequest(
            model=model_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
            tool_config=tool_config,
        )
        chat_details = await self._build_chat_details(request)
        if stream:
            return self._stream_chat_completion(request, chat_details)
        else:
            return await self._nonstream_chat_completion(request, chat_details)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest, details: ChatDetails
    ) -> AsyncGenerator[ChatCompletionResponseStreamChunk, None]:
        """
        Perform non-streaming chat completion using OCI Generative AI.
        """
        response = self._get_client().chat(details)
        stream = response.data

        async def _generate_and_convert_to_openai_compat():
            for chunk in stream.events():
                # {'index': 0, 'message': {'role': 'ASSISTANT', 'content': [{'type': 'TEXT', 'text': ' knowledge'}]}, 'pad': 'aaaaaaaa'}
                # {'message': {'role': 'ASSISTANT'}, 'finishReason': 'stop', 'pad': 'aaaaaaaaaaaa'}
                data = json.loads(chunk.data)
                finish_reason = data.get("finishReason", None)
                message_content = data.get("message", {}).get("content", [])
                text = ""
                if message_content:
                    text = message_content[0].get("text", "")
                choice = OpenAICompatCompletionChoice(finish_reason=finish_reason, text=text)
                yield OpenAICompatCompletionResponse(choices=[choice])

        stream = _generate_and_convert_to_openai_compat()
        async for chunk in process_chat_completion_stream_response(stream, request):
            yield chunk

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest, details: ChatDetails
    ) -> ChatCompletionResponse:
        """
        Perform streaming chat completion using OCI Generative AI.
        """
        response = self._get_client().chat(details)
        finish_reason = None
        message_content = ""
        if response.data.choices:
            finish_reason = response.data.choices[0].finish_reason
            message_content = (
                response.data.choices[0].message.content[0].text if response.data.choices[0].message.content else ""
            )
        choice = OpenAICompatCompletionChoice(
            finish_reason=finish_reason,
            text=message_content,
        )
        return process_chat_completion_response(OpenAICompatCompletionResponse(choices=[choice]), request)

    async def _build_openai_chat_details(self, params: dict) -> ChatDetails:
        messages = params.get("messages", [])
        system_messages = []
        user_messages = []
        structured_messages = []

        for msg in messages:
            if msg.get("role", "") == "system":
                system_messages.append(SystemMessage(content=[TextContent(text=msg.get("content", ""))]))
            else:
                user_messages.append(UserMessage(name="User", content=[TextContent(text=msg.get("content", ""))]))

        structured_messages.extend(system_messages)
        structured_messages.extend(user_messages)

        # Create OCI chat request
        chat_request = GenericChatRequest(
            api_format="GENERIC",
            messages=structured_messages,
            reasoning_effort=params.get("reasoning_effort"),
            verbosity=params.get("verbosity"),
            metadata=params.get("metadata"),
            is_stream=params.get("stream", False),
            stream_options=params.get("stream_options"),
            num_generations=params.get("n"),
            seed=params.get("seed"),
            is_echo=params.get("echo", False),
            top_k=params.get("top_k"),
            top_p=params.get("top_p"),
            temperature=params.get("temperature"),
            frequency_penalty=params.get("frequency_penalty"),
            presence_penalty=params.get("presence_penalty"),
            max_tokens=params.get("max_tokens"),
            max_completion_tokens=params.get("max_completion_tokens"),
            logit_bias=params.get("logit_bias"),
            # log_probs=params.get("log_probs", 0),
            # tool_choice=params.get("tool_choice", {}), # Unsupported
            # tools=params.get("tools", {}), # Unsupported
            # web_search_options=params.get("web_search_options", {}), # Unsupported
            # stop=params.get("stop", []),
        )

        model_id = self.get_provider_model_id(params.get("model", ""))
        if self.config.oci_serving_mode == OCI_SERVING_MODE_ON_DEMAND:
            serving_mode = OnDemandServingMode(
                serving_type="ON_DEMAND",
                model_id=model_id,
            )
        elif self.config.oci_serving_mode == OCI_SERVING_MODE_DEDICATED:
            serving_mode = DedicatedServingMode(serving_type="DEDICATED", model_id=model_id)

        chat_details = ChatDetails(
            compartment_id=self.config.oci_compartment_id, serving_mode=serving_mode, chat_request=chat_request
        )
        return chat_details

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        if self.model_store is None:
            raise ValueError("Model store is not initialized")
        model_obj = await self.model_store.get_model(params.model)
        request_params = await prepare_openai_completion_params(
            model=model_obj.provider_resource_id,
            messages=params.messages,
            frequency_penalty=params.frequency_penalty,
            function_call=params.function_call,
            functions=params.functions,
            logit_bias=params.logit_bias,
            logprobs=params.logprobs,
            max_completion_tokens=params.max_completion_tokens,
            max_tokens=params.max_tokens,
            n=params.n,
            parallel_tool_calls=params.parallel_tool_calls,
            presence_penalty=params.presence_penalty,
            response_format=params.response_format,
            seed=params.seed,
            stop=params.stop,
            stream=params.stream,
            stream_options=params.stream_options,
            temperature=params.temperature,
            tool_choice=params.tool_choice,
            tools=params.tools,
            top_logprobs=params.top_logprobs,
            top_p=params.top_p,
            user=params.user,
        )
        chat_details = await self._build_openai_chat_details(request_params)
        if request_params.get("stream", False):
            return self._stream_openai_chat_completion(chat_details)
        return await self._nonstream_openai_chat_completion(chat_details)

    async def _nonstream_openai_chat_completion(
        self,
        chat_details: ChatDetails,
    ) -> OpenAIChatCompletion:
        """Non-streaming OpenAI chat completion using OCI."""
        response: ChatResult = self._get_client().chat(chat_details)

        choice = OpenAIChoice(
            message=OpenAIAssistantMessageParam(
                role="assistant",
                content=response.data.chat_response.choices[0].message.content[0].text,
            ),
            finish_reason=response.data.chat_response.choices[0].finish_reason,
            index=response.data.chat_response.choices[0].index,
        )
        return OpenAIChatCompletion(
            id=str(response.data.chat_response.choices[0].index),
            choices=[choice],
            object="chat.completion",
            created=int(response.data.chat_response.time_created.timestamp()),
            model=response.data.model_id,
            usage=OpenAIChatCompletionUsage(
                prompt_tokens=response.data.chat_response.usage.prompt_tokens,
                completion_tokens=response.data.chat_response.usage.completion_tokens,
                total_tokens=response.data.chat_response.usage.total_tokens,
            ),
        )

    async def _stream_openai_chat_completion(
        self,
        chat_details: ChatDetails,
    ) -> AsyncIterator[OpenAIChatCompletionChunk]:
        """Streaming OpenAI chat completion using OCI."""
        response = self._get_client().chat(chat_details)
        stream = response.data

        i = 0
        for chunk in stream.events():
            i += 1

            data = json.loads(chunk.data)
            finish_reason = data.get("finishReason", "")
            message_content = data.get("message", {}).get("content", [])
            usage = data.get("usage", None)

            # Extract text content from the message content array
            text_content = ""
            if message_content:
                text_content = message_content[0].get("text", "")

            # Get model_id from the response data
            model_id = getattr(response.data, "model_id", None) or chat_details.serving_mode.model_id

            if usage:
                final_usage = OpenAIChatCompletionUsage(
                    prompt_tokens=usage.get("promptTokens", 0),
                    completion_tokens=usage.get("completionTokens", 0),
                    total_tokens=usage.get("totalTokens", 0),
                )
            else:
                final_usage = None
            yield OpenAIChatCompletionChunk(
                id=f"chunk-{i}",
                choices=[
                    OpenAIChunkChoice(
                        delta=OpenAIChoiceDelta(content=text_content),
                        finish_reason=finish_reason,
                        index=int(data.get("index", 0)),
                    )
                ],
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model_id,
                usage=final_usage,
            )

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion:
        raise NotImplementedError("OpenAI completion is not supported for OCI")

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError("OpenAI embeddings is not supported for OCI")
