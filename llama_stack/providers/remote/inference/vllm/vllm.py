# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any
from urllib.parse import urljoin

import httpx
from openai import APIConnectionError
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk as OpenAIChatCompletionChunk,
)

from llama_stack.apis.common.content_types import (
    TextDelta,
    ToolCallDelta,
    ToolCallParseStatus,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    GrammarResponseFormat,
    Inference,
    JsonSchemaResponseFormat,
    ModelStore,
    OpenAIChatCompletion,
    OpenAIMessageParam,
    OpenAIResponseFormatParam,
    ToolChoice,
    ToolDefinition,
)
from llama_stack.apis.models import Model, ModelType
from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import BuiltinTool, StopReason, ToolCall
from llama_stack.models.llama.sku_list import all_registered_models
from llama_stack.providers.datatypes import (
    HealthResponse,
    HealthStatus,
    ModelsProtocolPrivate,
)
from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
    build_hf_repo_model_entry,
)
from llama_stack.providers.utils.inference.openai_compat import (
    UnparseableToolCall,
    convert_message_to_openai_dict,
    convert_tool_call,
    get_sampling_options,
)
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import VLLMInferenceAdapterConfig

log = get_logger(name=__name__, category="inference::vllm")


def build_hf_repo_model_entries():
    return [
        build_hf_repo_model_entry(
            model.huggingface_repo,
            model.descriptor(),
        )
        for model in all_registered_models()
        if model.huggingface_repo
    ]


def _convert_to_vllm_tool_calls_in_response(
    tool_calls,
) -> list[ToolCall]:
    if not tool_calls:
        return []

    return [
        ToolCall(
            call_id=call.id,
            tool_name=call.function.name,
            arguments=call.function.arguments,
        )
        for call in tool_calls
    ]


def _convert_to_vllm_tools_in_request(tools: list[ToolDefinition]) -> list[dict]:
    compat_tools = []

    for tool in tools:
        # The tool.tool_name can be a str or a BuiltinTool enum. If
        # it's the latter, convert to a string.
        tool_name = tool.tool_name
        if isinstance(tool_name, BuiltinTool):
            tool_name = tool_name.value

        compat_tool = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool.description,
                "parameters": tool.input_schema
                or {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

        compat_tools.append(compat_tool)

    return compat_tools


def _convert_to_vllm_finish_reason(finish_reason: str) -> StopReason:
    return {
        "stop": StopReason.end_of_turn,
        "length": StopReason.out_of_tokens,
        "tool_calls": StopReason.end_of_message,
    }.get(finish_reason, StopReason.end_of_turn)


def _process_vllm_chat_completion_end_of_stream(
    finish_reason: str | None,
    last_chunk_content: str | None,
    current_event_type: ChatCompletionResponseEventType,
    tool_call_bufs: dict[str, UnparseableToolCall] | None = None,
) -> list[OpenAIChatCompletionChunk]:
    chunks = []

    if finish_reason is not None:
        stop_reason = _convert_to_vllm_finish_reason(finish_reason)
    else:
        stop_reason = StopReason.end_of_message

    tool_call_bufs = tool_call_bufs or {}
    for _index, tool_call_buf in sorted(tool_call_bufs.items()):
        args_str = tool_call_buf.arguments or "{}"
        try:
            chunks.append(
                ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=current_event_type,
                        delta=ToolCallDelta(
                            tool_call=ToolCall(
                                call_id=tool_call_buf.call_id,
                                tool_name=tool_call_buf.tool_name,
                                arguments=args_str,
                            ),
                            parse_status=ToolCallParseStatus.succeeded,
                        ),
                    )
                )
            )
        except Exception as e:
            log.warning(f"Failed to parse tool call buffer arguments: {args_str} \nError: {e}")

            chunks.append(
                ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            tool_call=str(tool_call_buf),
                            parse_status=ToolCallParseStatus.failed,
                        ),
                    )
                )
            )

    chunks.append(
        ChatCompletionResponseStreamChunk(
            event=ChatCompletionResponseEvent(
                event_type=ChatCompletionResponseEventType.complete,
                delta=TextDelta(text=last_chunk_content or ""),
                logprobs=None,
                stop_reason=stop_reason,
            )
        )
    )

    return chunks


async def _process_vllm_chat_completion_stream_response(
    stream: AsyncGenerator[OpenAIChatCompletionChunk, None],
) -> AsyncGenerator:
    yield ChatCompletionResponseStreamChunk(
        event=ChatCompletionResponseEvent(
            event_type=ChatCompletionResponseEventType.start,
            delta=TextDelta(text=""),
        )
    )
    event_type = ChatCompletionResponseEventType.progress
    tool_call_bufs: dict[str, UnparseableToolCall] = {}
    end_of_stream_processed = False

    async for chunk in stream:
        if not chunk.choices:
            log.warning("vLLM failed to generation any completions - check the vLLM server logs for an error.")
            return
        choice = chunk.choices[0]
        if choice.delta.tool_calls:
            for delta_tool_call in choice.delta.tool_calls:
                tool_call = convert_tool_call(delta_tool_call)
                if delta_tool_call.index not in tool_call_bufs:
                    tool_call_bufs[delta_tool_call.index] = UnparseableToolCall()
                tool_call_buf = tool_call_bufs[delta_tool_call.index]
                tool_call_buf.tool_name += str(tool_call.tool_name)
                tool_call_buf.call_id += tool_call.call_id
                tool_call_buf.arguments += (
                    tool_call.arguments if isinstance(tool_call.arguments, str) else json.dumps(tool_call.arguments)
                )
        if choice.finish_reason:
            chunks = _process_vllm_chat_completion_end_of_stream(
                finish_reason=choice.finish_reason,
                last_chunk_content=choice.delta.content,
                current_event_type=event_type,
                tool_call_bufs=tool_call_bufs,
            )
            for c in chunks:
                yield c
            end_of_stream_processed = True
        elif not choice.delta.tool_calls:
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=event_type,
                    delta=TextDelta(text=choice.delta.content or ""),
                    logprobs=None,
                )
            )
            event_type = ChatCompletionResponseEventType.progress

    if end_of_stream_processed:
        return

    # the stream ended without a chunk containing finish_reason - we have to generate the
    # respective completion chunks manually
    chunks = _process_vllm_chat_completion_end_of_stream(
        finish_reason=None, last_chunk_content=None, current_event_type=event_type, tool_call_bufs=tool_call_bufs
    )
    for c in chunks:
        yield c


class VLLMInferenceAdapter(OpenAIMixin, LiteLLMOpenAIMixin, Inference, ModelsProtocolPrivate):
    # automatically set by the resolver when instantiating the provider
    __provider_id__: str
    model_store: ModelStore | None = None

    def __init__(self, config: VLLMInferenceAdapterConfig) -> None:
        LiteLLMOpenAIMixin.__init__(
            self,
            model_entries=build_hf_repo_model_entries(),
            litellm_provider_name="vllm",
            api_key_from_config=config.api_token,
            provider_data_api_key_field="vllm_api_token",
            openai_compat_api_base=config.url,
        )
        self.register_helper = ModelRegistryHelper(build_hf_repo_model_entries())
        self.config = config

    get_api_key = LiteLLMOpenAIMixin.get_api_key

    def get_base_url(self) -> str:
        """Get the base URL from config."""
        if not self.config.url:
            raise ValueError("No base URL configured")
        return self.config.url

    async def initialize(self) -> None:
        if not self.config.url:
            raise ValueError(
                "You must provide a URL in run.yaml (or via the VLLM_URL environment variable) to use vLLM."
            )

    async def should_refresh_models(self) -> bool:
        # Strictly respecting the refresh_models directive
        return self.config.refresh_models

    async def list_models(self) -> list[Model] | None:
        models = []
        async for m in self.client.models.list():
            model_type = ModelType.llm  # unclear how to determine embedding vs. llm models
            models.append(
                Model(
                    identifier=m.id,
                    provider_resource_id=m.id,
                    provider_id=self.__provider_id__,
                    metadata={},
                    model_type=model_type,
                )
            )
        return models

    async def shutdown(self) -> None:
        pass

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def health(self) -> HealthResponse:
        """
        Performs a health check by verifying connectivity to the remote vLLM server.
        This method is used by the Provider API to verify
        that the service is running correctly.
        Uses the unauthenticated /health endpoint.
        Returns:

            HealthResponse: A dictionary containing the health status.
        """
        try:
            base_url = self.get_base_url()
            health_url = urljoin(base_url, "health")

            async with httpx.AsyncClient() as client:
                response = await client.get(health_url)
                response.raise_for_status()
                return HealthResponse(status=HealthStatus.OK)
        except Exception as e:
            return HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}")

    async def _get_model(self, model_id: str) -> Model:
        if not self.model_store:
            raise ValueError("Model store not set")
        return await self.model_store.get_model(model_id)

    def get_extra_client_params(self):
        return {"http_client": httpx.AsyncClient(verify=self.config.tls_verify)}

    async def register_model(self, model: Model) -> Model:
        try:
            model = await self.register_helper.register_model(model)
        except ValueError:
            pass  # Ignore statically unknown model, will check live listing
        try:
            res = self.client.models.list()
        except APIConnectionError as e:
            raise ValueError(
                f"Failed to connect to vLLM at {self.config.url}. Please check if vLLM is running and accessible at that URL."
            ) from e
        available_models = [m.id async for m in res]
        if model.provider_resource_id not in available_models:
            raise ValueError(
                f"Model {model.provider_resource_id} is not being served by vLLM. "
                f"Available models: {', '.join(available_models)}"
            )
        return model

    async def _get_params(self, request: ChatCompletionRequest) -> dict:
        options = get_sampling_options(request.sampling_params)
        if "max_tokens" not in options:
            options["max_tokens"] = self.config.max_tokens

        input_dict: dict[str, Any] = {}
        # Only include the 'tools' param if there is any. It can break things if an empty list is sent to the vLLM.
        if isinstance(request, ChatCompletionRequest) and request.tools:
            input_dict = {"tools": _convert_to_vllm_tools_in_request(request.tools)}

        input_dict["messages"] = [await convert_message_to_openai_dict(m, download=True) for m in request.messages]

        if fmt := request.response_format:
            if isinstance(fmt, JsonSchemaResponseFormat):
                input_dict["extra_body"] = {"guided_json": fmt.json_schema}
            elif isinstance(fmt, GrammarResponseFormat):
                raise NotImplementedError("Grammar response format not supported yet")
            else:
                raise ValueError(f"Unknown response format {fmt.type}")

        if request.logprobs and request.logprobs.top_k:
            input_dict["logprobs"] = request.logprobs.top_k

        return {
            "model": request.model,
            **input_dict,
            "stream": request.stream,
            **options,
        }

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
        max_tokens = max_tokens or self.config.max_tokens

        # This is to be consistent with OpenAI API and support vLLM <= v0.6.3
        # References:
        #   * https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
        #   * https://github.com/vllm-project/vllm/pull/10000
        if not tools and tool_choice is not None:
            tool_choice = ToolChoice.none.value

        return await super().openai_chat_completion(
            model=model,
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
