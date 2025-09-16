# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from openai import APIConnectionError, AsyncOpenAI
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk as OpenAIChatCompletionChunk,
)

from llama_stack.apis.common.content_types import (
    InterleavedContent,
    InterleavedContentItem,
    TextDelta,
    ToolCallDelta,
    ToolCallParseStatus,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    CompletionMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseStreamChunk,
    EmbeddingsResponse,
    EmbeddingTaskType,
    GrammarResponseFormat,
    Inference,
    JsonSchemaResponseFormat,
    LogProbConfig,
    Message,
    ModelStore,
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
from llama_stack.models.llama.datatypes import BuiltinTool, StopReason, ToolCall
from llama_stack.models.llama.sku_list import all_registered_models
from llama_stack.providers.datatypes import (
    HealthResponse,
    HealthStatus,
    ModelsProtocolPrivate,
)
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
    build_hf_repo_model_entry,
)
from llama_stack.providers.utils.inference.openai_compat import (
    UnparseableToolCall,
    convert_message_to_openai_dict,
    convert_tool_call,
    get_sampling_options,
    process_chat_completion_stream_response,
    process_completion_response,
    process_completion_stream_response,
)
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin
from llama_stack.providers.utils.inference.prompt_adapter import (
    completion_request_to_prompt,
    content_has_media,
    interleaved_content_as_str,
    request_has_media,
)

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
            arguments=json.loads(call.function.arguments),
            arguments_json=call.function.arguments,
        )
        for call in tool_calls
    ]


def _convert_to_vllm_tools_in_request(tools: list[ToolDefinition]) -> list[dict]:
    compat_tools = []

    for tool in tools:
        properties = {}
        compat_required = []
        if tool.parameters:
            for tool_key, tool_param in tool.parameters.items():
                properties[tool_key] = {"type": tool_param.param_type}
                if tool_param.description:
                    properties[tool_key]["description"] = tool_param.description
                if tool_param.default:
                    properties[tool_key]["default"] = tool_param.default
                if tool_param.required:
                    compat_required.append(tool_key)

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
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": compat_required,
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
            args = json.loads(args_str)
            chunks.append(
                ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=current_event_type,
                        delta=ToolCallDelta(
                            tool_call=ToolCall(
                                call_id=tool_call_buf.call_id,
                                tool_name=tool_call_buf.tool_name,
                                arguments=args,
                                arguments_json=args_str,
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


class VLLMInferenceAdapter(OpenAIMixin, Inference, ModelsProtocolPrivate):
    # automatically set by the resolver when instantiating the provider
    __provider_id__: str
    model_store: ModelStore | None = None

    def __init__(self, config: VLLMInferenceAdapterConfig) -> None:
        self.register_helper = ModelRegistryHelper(build_hf_repo_model_entries())
        self.config = config

    async def initialize(self) -> None:
        if not self.config.url:
            raise ValueError(
                "You must provide a URL in run.yaml (or via the VLLM_URL environment variable) to use vLLM."
            )

    async def should_refresh_models(self) -> bool:
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
        Returns:

            HealthResponse: A dictionary containing the health status.
        """
        try:
            _ = [m async for m in self.client.models.list()]  # Ensure the client is initialized
            return HealthResponse(status=HealthStatus.OK)
        except Exception as e:
            return HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}")

    async def _get_model(self, model_id: str) -> Model:
        if not self.model_store:
            raise ValueError("Model store not set")
        return await self.model_store.get_model(model_id)

    def get_api_key(self):
        return self.config.api_token

    def get_base_url(self):
        return self.config.url

    def get_extra_client_params(self):
        return {"http_client": httpx.AsyncClient(verify=self.config.tls_verify)}

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
        # This is to be consistent with OpenAI API and support vLLM <= v0.6.3
        # References:
        #   * https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
        #   * https://github.com/vllm-project/vllm/pull/10000
        if not tools and tool_config is not None:
            tool_config.tool_choice = ToolChoice.none
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
            return self._stream_chat_completion(request, self.client)
        else:
            return await self._nonstream_chat_completion(request, self.client)

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest, client: AsyncOpenAI
    ) -> ChatCompletionResponse:
        params = await self._get_params(request)
        r = await client.chat.completions.create(**params)
        choice = r.choices[0]
        result = ChatCompletionResponse(
            completion_message=CompletionMessage(
                content=choice.message.content or "",
                stop_reason=_convert_to_vllm_finish_reason(choice.finish_reason),
                tool_calls=_convert_to_vllm_tool_calls_in_response(choice.message.tool_calls),
            ),
            logprobs=None,
        )
        return result

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest, client: AsyncOpenAI
    ) -> AsyncGenerator[ChatCompletionResponseStreamChunk, None]:
        params = await self._get_params(request)

        stream = await client.chat.completions.create(**params)
        if request.tools:
            res = _process_vllm_chat_completion_stream_response(stream)
        else:
            res = process_chat_completion_stream_response(stream, request)
        async for chunk in res:
            yield chunk

    async def _nonstream_completion(self, request: CompletionRequest) -> CompletionResponse:
        assert self.client is not None
        params = await self._get_params(request)
        r = await self.client.completions.create(**params)
        return process_completion_response(r)

    async def _stream_completion(
        self, request: CompletionRequest
    ) -> AsyncGenerator[CompletionResponseStreamChunk, None]:
        assert self.client is not None
        params = await self._get_params(request)

        stream = await self.client.completions.create(**params)
        async for chunk in process_completion_stream_response(stream):
            yield chunk

    async def register_model(self, model: Model) -> Model:
        try:
            model = await self.register_helper.register_model(model)
        except ValueError:
            pass  # Ignore statically unknown model, will check live listing
        try:
            res = await self.client.models.list()
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

    async def _get_params(self, request: ChatCompletionRequest | CompletionRequest) -> dict:
        options = get_sampling_options(request.sampling_params)
        if "max_tokens" not in options:
            options["max_tokens"] = self.config.max_tokens

        input_dict: dict[str, Any] = {}
        # Only include the 'tools' param if there is any. It can break things if an empty list is sent to the vLLM.
        if isinstance(request, ChatCompletionRequest) and request.tools:
            input_dict = {"tools": _convert_to_vllm_tools_in_request(request.tools)}

        if isinstance(request, ChatCompletionRequest):
            input_dict["messages"] = [await convert_message_to_openai_dict(m, download=True) for m in request.messages]
        else:
            assert not request_has_media(request), "vLLM does not support media for Completion requests"
            input_dict["prompt"] = await completion_request_to_prompt(request)

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

    async def embeddings(
        self,
        model_id: str,
        contents: list[str] | list[InterleavedContentItem],
        text_truncation: TextTruncation | None = TextTruncation.none,
        output_dimension: int | None = None,
        task_type: EmbeddingTaskType | None = None,
    ) -> EmbeddingsResponse:
        model = await self._get_model(model_id)

        kwargs = {}
        assert model.model_type == ModelType.embedding
        assert model.metadata.get("embedding_dimension")
        kwargs["dimensions"] = model.metadata.get("embedding_dimension")
        assert all(not content_has_media(content) for content in contents), "VLLM does not support media for embeddings"
        response = await self.client.embeddings.create(
            model=model.provider_resource_id,
            input=[interleaved_content_as_str(content) for content in contents],
            **kwargs,
        )

        embeddings = [data.embedding for data in response.data]
        return EmbeddingsResponse(embeddings=embeddings)
