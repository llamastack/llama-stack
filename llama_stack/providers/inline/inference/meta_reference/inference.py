# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import os
import sys
from collections.abc import AsyncGenerator

from pydantic import BaseModel
from termcolor import cprint

from llama_stack.apis.common.content_types import (
    TextDelta,
    ToolCallDelta,
    ToolCallParseStatus,
)
from llama_stack.apis.inference import (
    BatchChatCompletionResponse,
    BatchCompletionResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    CompletionMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseStreamChunk,
    InferenceProvider,
    InterleavedContent,
    LogProbConfig,
    Message,
    ResponseFormat,
    SamplingParams,
    StopReason,
    TokenLogProbs,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
    UserMessage,
)
from llama_stack.apis.models import Model, ModelType
from llama_stack.log import get_logger
from llama_stack.models.llama.llama3.chat_format import ChatFormat as Llama3ChatFormat
from llama_stack.models.llama.llama3.tokenizer import Tokenizer as Llama3Tokenizer
from llama_stack.models.llama.llama4.chat_format import ChatFormat as Llama4ChatFormat
from llama_stack.models.llama.llama4.tokenizer import Tokenizer as Llama4Tokenizer
from llama_stack.models.llama.sku_list import resolve_model
from llama_stack.models.llama.sku_types import ModelFamily
from llama_stack.providers.datatypes import ModelsProtocolPrivate
from llama_stack.providers.utils.inference.embedding_mixin import (
    SentenceTransformerEmbeddingMixin,
)
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
    build_hf_repo_model_entry,
)
from llama_stack.providers.utils.inference.openai_compat import (
    OpenAIChatCompletionToLlamaStackMixin,
    OpenAICompletionToLlamaStackMixin,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    augment_content_with_response_format_prompt,
    chat_completion_request_to_messages,
    convert_request_to_raw,
)

from .config import MetaReferenceInferenceConfig
from .generators import LlamaGenerator
from .model_parallel import LlamaModelParallelGenerator

log = get_logger(__name__, category="inference")
# there's a single model parallel process running serving the model. for now,
# we don't support multiple concurrent requests to this process.
SEMAPHORE = asyncio.Semaphore(1)


def llama_builder_fn(config: MetaReferenceInferenceConfig, model_id: str, llama_model: Model) -> LlamaGenerator:
    return LlamaGenerator(config, model_id, llama_model)


class MetaReferenceInferenceImpl(
    OpenAICompletionToLlamaStackMixin,
    OpenAIChatCompletionToLlamaStackMixin,
    SentenceTransformerEmbeddingMixin,
    InferenceProvider,
    ModelsProtocolPrivate,
):
    def __init__(self, config: MetaReferenceInferenceConfig) -> None:
        self.config = config
        self.model_id = None
        self.llama_model = None

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        if self.config.create_distributed_process_group:
            self.generator.stop()

    async def should_refresh_models(self) -> bool:
        return False

    async def list_models(self) -> list[Model] | None:
        return None

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def register_model(self, model: Model) -> Model:
        llama_model = (
            resolve_model(model.metadata["llama_model"])
            if "llama_model" in model.metadata
            else resolve_model(model.identifier)
        )
        if llama_model is None:
            raise ValueError(
                "Please make sure your llama_model in model metadata or model identifier is in Llama SKU list"
            )

        self.model_registry_helper = ModelRegistryHelper(
            [
                build_hf_repo_model_entry(
                    llama_model.descriptor(),
                    llama_model.core_model_id.value,
                )
            ],
        )
        model = await self.model_registry_helper.register_model(model)

        if model.model_type == ModelType.embedding:
            self._load_sentence_transformer_model(model.provider_resource_id)

        # TODO: what is this?! you can't really specify skipping via model metadata
        # kill this madness
        if "skip_load" in model.metadata and model.metadata["skip_load"]:
            return model

        await self.load_model(model.identifier, llama_model)
        return model

    async def load_model(self, model_id, llama_model) -> None:
        log.info(f"Loading model `{model_id}`")

        builder_params = [self.config, model_id, llama_model]

        if self.config.create_distributed_process_group:
            self.generator = LlamaModelParallelGenerator(
                model_parallel_size=self.config.model_parallel_size or llama_model.pth_file_count,
                builder_fn=llama_builder_fn,
                builder_params=builder_params,
                formatter=(
                    Llama4ChatFormat(Llama4Tokenizer.get_instance())
                    if llama_model.model_family == ModelFamily.llama4
                    else Llama3ChatFormat(Llama3Tokenizer.get_instance())
                ),
            )
            self.generator.start()
        else:
            self.generator = llama_builder_fn(*builder_params)

        self.model_id = model_id
        self.llama_model = llama_model

        log.info("Warming up...")
        await self.completion(
            model_id=model_id,
            content="Hello, world!",
            sampling_params=SamplingParams(max_tokens=10),
        )
        await self.chat_completion(
            model_id=model_id,
            messages=[UserMessage(content="Hi how are you?")],
            sampling_params=SamplingParams(max_tokens=20),
        )
        log.info("Warmed up!")

    def check_model(self, request) -> None:
        if self.model_id is None or self.llama_model is None:
            raise RuntimeError(
                "No avaible model yet, please register your requested model or add your model in the resouces first"
            )
        elif request.model != self.model_id:
            raise RuntimeError(f"Model mismatch: request model: {request.model} != loaded model: {self.model_id}")

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
    ) -> CompletionResponse | CompletionResponseStreamChunk:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if logprobs:
            assert logprobs.top_k == 1, f"Unexpected top_k={logprobs.top_k}"

        content = augment_content_with_response_format_prompt(response_format, content)
        request = CompletionRequest(
            model=model_id,
            content=content,
            sampling_params=sampling_params,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )
        self.check_model(request)
        request = await convert_request_to_raw(request)

        if request.stream:
            return self._stream_completion(request)
        else:
            results = await self._nonstream_completion([request])
            return results[0]

    async def batch_completion(
        self,
        model_id: str,
        content_batch: list[InterleavedContent],
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
    ) -> BatchCompletionResponse:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if logprobs:
            assert logprobs.top_k == 1, f"Unexpected top_k={logprobs.top_k}"

        content_batch = [
            augment_content_with_response_format_prompt(response_format, content) for content in content_batch
        ]

        request_batch = []
        for content in content_batch:
            request = CompletionRequest(
                model=model_id,
                content=content,
                sampling_params=sampling_params,
                response_format=response_format,
                stream=stream,
                logprobs=logprobs,
            )
            self.check_model(request)
            request = await convert_request_to_raw(request)
            request_batch.append(request)

        results = await self._nonstream_completion(request_batch)
        return BatchCompletionResponse(batch=results)

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        tokenizer = self.generator.formatter.tokenizer

        def impl():
            stop_reason = None

            for token_results in self.generator.completion([request]):
                token_result = token_results[0]
                if token_result.token == tokenizer.eot_id:
                    stop_reason = StopReason.end_of_turn
                    text = ""
                elif token_result.token == tokenizer.eom_id:
                    stop_reason = StopReason.end_of_message
                    text = ""
                else:
                    text = token_result.text

                logprobs = None
                if stop_reason is None:
                    if request.logprobs:
                        assert len(token_result.logprobs) == 1

                        logprobs = [TokenLogProbs(logprobs_by_token={token_result.text: token_result.logprobs[0]})]

                yield CompletionResponseStreamChunk(
                    delta=text,
                    stop_reason=stop_reason,
                    logprobs=logprobs if request.logprobs else None,
                )

            if stop_reason is None:
                yield CompletionResponseStreamChunk(
                    delta="",
                    stop_reason=StopReason.out_of_tokens,
                )

        if self.config.create_distributed_process_group:
            async with SEMAPHORE:
                for x in impl():
                    yield x
        else:
            for x in impl():
                yield x

    async def _nonstream_completion(self, request_batch: list[CompletionRequest]) -> list[CompletionResponse]:
        tokenizer = self.generator.formatter.tokenizer

        first_request = request_batch[0]

        class ItemState(BaseModel):
            tokens: list[int] = []
            logprobs: list[TokenLogProbs] = []
            stop_reason: StopReason | None = None
            finished: bool = False

        def impl():
            states = [ItemState() for _ in request_batch]

            results = []
            for token_results in self.generator.completion(request_batch):
                for result in token_results:
                    idx = result.batch_idx
                    state = states[idx]
                    if state.finished or result.ignore_token:
                        continue

                    state.finished = result.finished
                    if first_request.logprobs:
                        state.logprobs.append(TokenLogProbs(logprobs_by_token={result.text: result.logprobs[0]}))

                    state.tokens.append(result.token)
                    if result.token == tokenizer.eot_id:
                        state.stop_reason = StopReason.end_of_turn
                    elif result.token == tokenizer.eom_id:
                        state.stop_reason = StopReason.end_of_message

            for state in states:
                if state.stop_reason is None:
                    state.stop_reason = StopReason.out_of_tokens

                if state.tokens[-1] in self.generator.formatter.tokenizer.stop_tokens:
                    state.tokens = state.tokens[:-1]
                content = self.generator.formatter.tokenizer.decode(state.tokens)
                results.append(
                    CompletionResponse(
                        content=content,
                        stop_reason=state.stop_reason,
                        logprobs=state.logprobs if first_request.logprobs else None,
                    )
                )

            return results

        if self.config.create_distributed_process_group:
            async with SEMAPHORE:
                return impl()
        else:
            return impl()

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
        if sampling_params is None:
            sampling_params = SamplingParams()
        if logprobs:
            assert logprobs.top_k == 1, f"Unexpected top_k={logprobs.top_k}"

        # wrapper request to make it easier to pass around (internal only, not exposed to API)
        request = ChatCompletionRequest(
            model=model_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
            tool_config=tool_config or ToolConfig(),
        )
        self.check_model(request)

        # augment and rewrite messages depending on the model
        request.messages = chat_completion_request_to_messages(request, self.llama_model.core_model_id.value)
        # download media and convert to raw content so we can send it to the model
        request = await convert_request_to_raw(request)

        if self.config.create_distributed_process_group:
            if SEMAPHORE.locked():
                raise RuntimeError("Only one concurrent request is supported")

        if request.stream:
            return self._stream_chat_completion(request)
        else:
            results = await self._nonstream_chat_completion([request])
            return results[0]

    async def batch_chat_completion(
        self,
        model_id: str,
        messages_batch: list[list[Message]],
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        tools: list[ToolDefinition] | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
        tool_config: ToolConfig | None = None,
    ) -> BatchChatCompletionResponse:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if logprobs:
            assert logprobs.top_k == 1, f"Unexpected top_k={logprobs.top_k}"

        # wrapper request to make it easier to pass around (internal only, not exposed to API)
        request_batch = []
        for messages in messages_batch:
            request = ChatCompletionRequest(
                model=model_id,
                messages=messages,
                sampling_params=sampling_params,
                tools=tools or [],
                response_format=response_format,
                logprobs=logprobs,
                tool_config=tool_config or ToolConfig(),
            )
            self.check_model(request)

            # augment and rewrite messages depending on the model
            request.messages = chat_completion_request_to_messages(request, self.llama_model.core_model_id.value)
            # download media and convert to raw content so we can send it to the model
            request = await convert_request_to_raw(request)
            request_batch.append(request)

        if self.config.create_distributed_process_group:
            if SEMAPHORE.locked():
                raise RuntimeError("Only one concurrent request is supported")

        results = await self._nonstream_chat_completion(request_batch)
        return BatchChatCompletionResponse(batch=results)

    async def _nonstream_chat_completion(
        self, request_batch: list[ChatCompletionRequest]
    ) -> list[ChatCompletionResponse]:
        tokenizer = self.generator.formatter.tokenizer

        first_request = request_batch[0]

        class ItemState(BaseModel):
            tokens: list[int] = []
            logprobs: list[TokenLogProbs] = []
            stop_reason: StopReason | None = None
            finished: bool = False

        def impl():
            states = [ItemState() for _ in request_batch]

            for token_results in self.generator.chat_completion(request_batch):
                first = token_results[0]
                if not first.finished and not first.ignore_token:
                    if os.environ.get("LLAMA_MODELS_DEBUG", "0") in ("1", "2"):
                        cprint(first.text, color="cyan", end="", file=sys.stderr)
                    if os.environ.get("LLAMA_MODELS_DEBUG", "0") == "2":
                        cprint(f"<{first.token}>", color="magenta", end="", file=sys.stderr)

                for result in token_results:
                    idx = result.batch_idx
                    state = states[idx]
                    if state.finished or result.ignore_token:
                        continue

                    state.finished = result.finished
                    if first_request.logprobs:
                        state.logprobs.append(TokenLogProbs(logprobs_by_token={result.text: result.logprobs[0]}))

                    state.tokens.append(result.token)
                    if result.token == tokenizer.eot_id:
                        state.stop_reason = StopReason.end_of_turn
                    elif result.token == tokenizer.eom_id:
                        state.stop_reason = StopReason.end_of_message

            results = []
            for state in states:
                if state.stop_reason is None:
                    state.stop_reason = StopReason.out_of_tokens

                raw_message = self.generator.formatter.decode_assistant_message(state.tokens, state.stop_reason)
                results.append(
                    ChatCompletionResponse(
                        completion_message=CompletionMessage(
                            content=raw_message.content,
                            stop_reason=raw_message.stop_reason,
                            tool_calls=raw_message.tool_calls,
                        ),
                        logprobs=state.logprobs if first_request.logprobs else None,
                    )
                )

            return results

        if self.config.create_distributed_process_group:
            async with SEMAPHORE:
                return impl()
        else:
            return impl()

    async def _stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        tokenizer = self.generator.formatter.tokenizer

        def impl():
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.start,
                    delta=TextDelta(text=""),
                )
            )

            tokens = []
            logprobs = []
            stop_reason = None
            ipython = False

            for token_results in self.generator.chat_completion([request]):
                token_result = token_results[0]
                if os.environ.get("LLAMA_MODELS_DEBUG", "0") == "1":
                    cprint(token_result.text, color="cyan", end="", file=sys.stderr)
                if os.environ.get("LLAMA_MODELS_DEBUG", "0") == "2":
                    cprint(f"<{token_result.token}>", color="magenta", end="", file=sys.stderr)

                if token_result.token == tokenizer.eot_id:
                    stop_reason = StopReason.end_of_turn
                    text = ""
                elif token_result.token == tokenizer.eom_id:
                    stop_reason = StopReason.end_of_message
                    text = ""
                else:
                    text = token_result.text

                if request.logprobs:
                    assert len(token_result.logprobs) == 1

                    logprobs.append(TokenLogProbs(logprobs_by_token={token_result.text: token_result.logprobs[0]}))

                tokens.append(token_result.token)

                if not ipython and token_result.text.startswith("<|python_tag|>"):
                    ipython = True
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=ToolCallDelta(
                                tool_call="",
                                parse_status=ToolCallParseStatus.started,
                            ),
                        )
                    )
                    continue

                if token_result.token == tokenizer.eot_id:
                    stop_reason = StopReason.end_of_turn
                    text = ""
                elif token_result.token == tokenizer.eom_id:
                    stop_reason = StopReason.end_of_message
                    text = ""
                else:
                    text = token_result.text

                if ipython:
                    delta = ToolCallDelta(
                        tool_call=text,
                        parse_status=ToolCallParseStatus.in_progress,
                    )
                else:
                    delta = TextDelta(text=text)

                if stop_reason is None:
                    if request.logprobs:
                        assert len(token_result.logprobs) == 1

                        logprobs.append(TokenLogProbs(logprobs_by_token={token_result.text: token_result.logprobs[0]}))
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=delta,
                            stop_reason=stop_reason,
                            logprobs=logprobs if request.logprobs else None,
                        )
                    )

            if stop_reason is None:
                stop_reason = StopReason.out_of_tokens

            message = self.generator.formatter.decode_assistant_message(tokens, stop_reason)

            parsed_tool_calls = len(message.tool_calls) > 0
            if ipython and not parsed_tool_calls:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            tool_call="",
                            parse_status=ToolCallParseStatus.failed,
                        ),
                        stop_reason=stop_reason,
                    )
                )

            for tool_call in message.tool_calls:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            tool_call=tool_call,
                            parse_status=ToolCallParseStatus.succeeded,
                        ),
                        stop_reason=stop_reason,
                    )
                )

            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.complete,
                    delta=TextDelta(text=""),
                    stop_reason=stop_reason,
                )
            )

        if self.config.create_distributed_process_group:
            async with SEMAPHORE:
                for x in impl():
                    yield x
        else:
            for x in impl():
                yield x
