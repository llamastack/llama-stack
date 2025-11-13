# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import time
import uuid
from collections.abc import AsyncIterator

from llama_stack.apis.inference import (
    InferenceProvider,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionUsage,
    OpenAIChoice,
    OpenAICompletionRequestWithExtraBody,
    OpenAIUserMessageParam,
    ToolChoice,
)
from llama_stack.apis.inference.inference import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAICompletion,
)
from llama_stack.apis.models import Model, ModelType
from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import RawMessage, RawTextItem, ToolDefinition
from llama_stack.models.llama.llama3.chat_format import ChatFormat as Llama3ChatFormat
from llama_stack.models.llama.llama3.prompt_templates import (
    JsonCustomToolGenerator,
    SystemDefaultGenerator,
)
from llama_stack.models.llama.llama3.tokenizer import Tokenizer as Llama3Tokenizer
from llama_stack.models.llama.llama4.chat_format import ChatFormat as Llama4ChatFormat
from llama_stack.models.llama.llama4.prompt_templates.system_prompts import (
    PythonListCustomToolGenerator as PythonListCustomToolGeneratorLlama4,
)
from llama_stack.models.llama.llama4.tokenizer import Tokenizer as Llama4Tokenizer
from llama_stack.models.llama.sku_list import resolve_model
from llama_stack.models.llama.sku_types import ModelFamily, is_multimodal
from llama_stack.providers.datatypes import ModelsProtocolPrivate
from llama_stack.providers.utils.inference.embedding_mixin import (
    SentenceTransformerEmbeddingMixin,
)
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
    build_hf_repo_model_entry,
)

from .config import MetaReferenceInferenceConfig
from .generators import LlamaGenerator
from .model_parallel import LlamaModelParallelGenerator

log = get_logger(__name__, category="inference")
# there's a single model parallel process running serving the model. for now,
# we don't support multiple concurrent requests to this process.
SEMAPHORE = asyncio.Semaphore(1)


def _convert_openai_tool_to_tool_definition(tool) -> ToolDefinition:
    """Convert OpenAI tool format to ToolDefinition format."""
    # OpenAI tools have function.name and function.parameters
    return ToolDefinition(
        tool_name=tool.function.name,
        description=tool.function.description or "",
        parameters=tool.function.parameters or {},
    )


def _get_tool_choice_prompt(tool_choice, tools) -> str:
    """Generate prompt text for tool_choice behavior."""
    if not tool_choice or tool_choice == ToolChoice.auto or tool_choice == "auto":
        return ""
    elif tool_choice == ToolChoice.required or tool_choice == "required":
        return "You MUST use one of the provided functions/tools to answer the user query."
    elif tool_choice == ToolChoice.none or tool_choice == "none":
        return ""
    else:
        # Specific tool specified
        return f"You MUST use the tool `{tool_choice}` to answer the user query."


def _raw_content_as_str(content) -> str:
    """Convert RawContent to string for system messages."""
    if isinstance(content, str):
        return content
    elif isinstance(content, RawTextItem):
        return content.text
    elif isinstance(content, list):
        return "\n".join(_raw_content_as_str(c) for c in content)
    else:
        return "<media>"


def _augment_raw_messages_for_tools_llama_3_1(
    raw_messages: list[RawMessage],
    tools: list,
    tool_choice,
) -> list[RawMessage]:
    """Augment raw messages with tool definitions for Llama 3.1 style models."""
    messages = raw_messages.copy()
    existing_system_message = None
    if messages and messages[0].role == "system":
        existing_system_message = messages.pop(0)

    sys_content = ""

    # Add tool definitions first (if present)
    if tools:
        # Convert OpenAI tools to ToolDefinitions
        tool_definitions = [_convert_openai_tool_to_tool_definition(t) for t in tools]

        # For OpenAI format, all tools are custom (have string names)
        tool_gen = JsonCustomToolGenerator()
        tool_template = tool_gen.gen(tool_definitions)
        sys_content += tool_template.render()
        sys_content += "\n"

    # Add default system prompt
    default_gen = SystemDefaultGenerator()
    default_template = default_gen.gen()
    sys_content += default_template.render()

    # Add existing system message if present
    if existing_system_message:
        sys_content += "\n" + _raw_content_as_str(existing_system_message.content)

    # Add tool choice prompt if needed
    if tool_choice_prompt := _get_tool_choice_prompt(tool_choice, tools):
        sys_content += "\n" + tool_choice_prompt

    # Create new system message
    new_system_message = RawMessage(
        role="system",
        content=[RawTextItem(text=sys_content.strip())],
    )

    return [new_system_message] + messages


def _augment_raw_messages_for_tools_llama_4(
    raw_messages: list[RawMessage],
    tools: list,
    tool_choice,
) -> list[RawMessage]:
    """Augment raw messages with tool definitions for Llama 4/3.2/3.3 style models."""
    messages = raw_messages.copy()
    existing_system_message = None
    if messages and messages[0].role == "system":
        existing_system_message = messages.pop(0)

    sys_content = ""

    # Add tool definitions if present
    if tools:
        # Convert OpenAI tools to ToolDefinitions
        tool_definitions = [_convert_openai_tool_to_tool_definition(t) for t in tools]

        # Use python_list format for Llama 4
        tool_gen = PythonListCustomToolGeneratorLlama4()
        system_prompt = None
        if existing_system_message:
            system_prompt = _raw_content_as_str(existing_system_message.content)

        tool_template = tool_gen.gen(tool_definitions, system_prompt)
        sys_content = tool_template.render()
    elif existing_system_message:
        # No tools, just use existing system message
        sys_content = _raw_content_as_str(existing_system_message.content)

    # Add tool choice prompt if needed
    if tool_choice_prompt := _get_tool_choice_prompt(tool_choice, tools):
        sys_content += "\n" + tool_choice_prompt

    if sys_content:
        new_system_message = RawMessage(
            role="system",
            content=[RawTextItem(text=sys_content.strip())],
        )
        return [new_system_message] + messages

    return messages


def augment_raw_messages_for_tools(
    raw_messages: list[RawMessage],
    params: OpenAIChatCompletionRequestWithExtraBody,
    llama_model,
) -> list[RawMessage]:
    """Augment raw messages with tool definitions based on model family."""
    if not params.tools:
        return raw_messages

    # Determine augmentation strategy based on model family
    if llama_model.model_family == ModelFamily.llama3_1 or (
        llama_model.model_family == ModelFamily.llama3_2 and is_multimodal(llama_model.core_model_id)
    ):
        # Llama 3.1 and Llama 3.2 multimodal use JSON format
        return _augment_raw_messages_for_tools_llama_3_1(
            raw_messages,
            params.tools,
            params.tool_choice,
        )
    elif llama_model.model_family in (
        ModelFamily.llama3_2,
        ModelFamily.llama3_3,
        ModelFamily.llama4,
    ):
        # Llama 3.2/3.3/4 use python_list format
        return _augment_raw_messages_for_tools_llama_4(
            raw_messages,
            params.tools,
            params.tool_choice,
        )
    else:
        # Default to Llama 3.1 style
        return _augment_raw_messages_for_tools_llama_3_1(
            raw_messages,
            params.tools,
            params.tool_choice,
        )


def llama_builder_fn(config: MetaReferenceInferenceConfig, model_id: str, llama_model: Model) -> LlamaGenerator:
    return LlamaGenerator(config, model_id, llama_model)


class MetaReferenceInferenceImpl(
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

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion:
        raise NotImplementedError("OpenAI completion not supported by meta reference provider")

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

        await self.openai_chat_completion(
            params=OpenAIChatCompletionRequestWithExtraBody(
                model=model_id,
                messages=[OpenAIUserMessageParam(role="user", content="Hi how are you?")],
                max_tokens=20,
            )
        )
        log.info("Warmed up!")

    def check_model(self, request) -> None:
        if self.model_id is None or self.llama_model is None:
            raise RuntimeError(
                "No available model yet, please register your requested model or add your model in the resources first"
            )
        elif request.model != self.model_id:
            raise RuntimeError(f"Model mismatch: request model: {request.model} != loaded model: {self.model_id}")

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        self.check_model(params)

        # Convert OpenAI messages to RawMessages
        from llama_stack.models.llama.datatypes import StopReason
        from llama_stack.providers.utils.inference.prompt_adapter import (
            convert_openai_message_to_raw_message,
            decode_assistant_message,
        )

        raw_messages = [await convert_openai_message_to_raw_message(msg) for msg in params.messages]

        # Augment messages with tool definitions if tools are present
        raw_messages = augment_raw_messages_for_tools(raw_messages, params, self.llama_model)

        # Call generator's chat_completion method (works for both single-GPU and model-parallel)
        if isinstance(self.generator, LlamaGenerator):
            generator = self.generator.chat_completion(params, raw_messages)
        else:
            # Model parallel: submit task to process group
            generator = self.generator.group.run_inference(("chat_completion", [params, raw_messages]))

        # Check if streaming is requested
        if params.stream:
            return self._stream_chat_completion(generator, params)

        # Non-streaming: collect all generated text
        generated_text = ""
        for result_batch in generator:
            for result in result_batch:
                if not result.ignore_token and result.source == "output":
                    generated_text += result.text

        # Decode assistant message to extract tool calls and determine stop_reason
        # Default to end_of_turn if generation completed normally
        decoded_message = decode_assistant_message(generated_text, StopReason.end_of_turn)

        # Convert tool calls to OpenAI format
        openai_tool_calls = None
        if decoded_message.tool_calls:
            from llama_stack.apis.inference import (
                OpenAIChatCompletionToolCall,
                OpenAIChatCompletionToolCallFunction,
            )

            openai_tool_calls = [
                OpenAIChatCompletionToolCall(
                    # generate a uuid for the call id. This is the only inline provider that does this, so need to get creative.
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    type="function",
                    function=OpenAIChatCompletionToolCallFunction(
                        name=str(tc.tool_name),
                        arguments=tc.arguments,
                    ),
                )
                for tc in decoded_message.tool_calls
            ]

        # Determine finish_reason based on whether tool calls are present
        finish_reason = "tool_calls" if openai_tool_calls else "stop"

        # Extract content from decoded message
        content = ""
        if isinstance(decoded_message.content, str):
            content = decoded_message.content
        elif isinstance(decoded_message.content, list):
            for item in decoded_message.content:
                if isinstance(item, RawTextItem):
                    content += item.text

        # Create OpenAI response
        # generate a uuid for the call id. This is the only inline provider that does this, so need to get creative.
        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        return OpenAIChatCompletion(
            id=response_id,
            object="chat.completion",
            created=created,
            model=params.model,
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIAssistantMessageParam(
                        role="assistant",
                        content=content,
                        tool_calls=openai_tool_calls,
                    ),
                    finish_reason=finish_reason,
                    logprobs=None,
                )
            ],
            usage=OpenAIChatCompletionUsage(
                prompt_tokens=0,  # TODO: calculate properly
                completion_tokens=0,  # TODO: calculate properly
                total_tokens=0,  # TODO: calculate properly
            ),
        )

    async def _stream_chat_completion(
        self,
        generator,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> AsyncIterator[OpenAIChatCompletionChunk]:
        """Stream chat completion chunks as they're generated."""
        from llama_stack.apis.inference import (
            OpenAIChatCompletionChunk,
            OpenAIChatCompletionToolCall,
            OpenAIChatCompletionToolCallFunction,
            OpenAIChoiceDelta,
            OpenAIChunkChoice,
        )
        from llama_stack.models.llama.datatypes import StopReason
        from llama_stack.providers.utils.inference.prompt_adapter import decode_assistant_message

        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        generated_text = ""

        # Yield chunks as tokens are generated
        for result_batch in generator:
            for result in result_batch:
                if result.ignore_token or result.source != "output":
                    continue

                generated_text += result.text

                # Yield delta chunk with the new text
                chunk = OpenAIChatCompletionChunk(
                    id=response_id,
                    object="chat.completion.chunk",
                    created=created,
                    model=params.model,
                    choices=[
                        OpenAIChunkChoice(
                            index=0,
                            delta=OpenAIChoiceDelta(
                                role="assistant",
                                content=result.text,
                            ),
                            finish_reason="",
                            logprobs=None,
                        )
                    ],
                )
                yield chunk

        # After generation completes, decode the full message to extract tool calls
        decoded_message = decode_assistant_message(generated_text, StopReason.end_of_turn)

        # If tool calls are present, yield a final chunk with tool_calls
        if decoded_message.tool_calls:
            openai_tool_calls = [
                OpenAIChatCompletionToolCall(
                    # generate a uuid for the call id. This is the only inline provider that does this, so need to get creative.
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    type="function",
                    function=OpenAIChatCompletionToolCallFunction(
                        name=str(tc.tool_name),
                        arguments=tc.arguments,
                    ),
                )
                for tc in decoded_message.tool_calls
            ]

            # Yield chunk with tool_calls
            chunk = OpenAIChatCompletionChunk(
                id=response_id,
                object="chat.completion.chunk",
                created=created,
                model=params.model,
                choices=[
                    OpenAIChunkChoice(
                        index=0,
                        delta=OpenAIChoiceDelta(
                            role="assistant",
                            tool_calls=openai_tool_calls,
                        ),
                        finish_reason="",
                        logprobs=None,
                    )
                ],
            )
            yield chunk

            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"

        # Yield final chunk with finish_reason
        final_chunk = OpenAIChatCompletionChunk(
            id=response_id,
            object="chat.completion.chunk",
            created=created,
            model=params.model,
            choices=[
                OpenAIChunkChoice(
                    index=0,
                    delta=OpenAIChoiceDelta(),
                    finish_reason=finish_reason,
                    logprobs=None,
                )
            ],
        )
        yield final_chunk
