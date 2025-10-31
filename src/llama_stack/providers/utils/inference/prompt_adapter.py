# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import base64
import io
import json
import re
from typing import Any

import httpx
from PIL import Image as PIL_Image

from llama_stack.apis.common.content_types import (
    ImageContentItem,
    InterleavedContent,
    InterleavedContentItem,
    TextContentItem,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    CompletionRequest,
    Message,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIFile,
    ResponseFormat,
    ResponseFormatType,
    SystemMessage,
    SystemMessageBehavior,
    ToolChoice,
    ToolDefinition,
    UserMessage,
)
from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import (
    RawContent,
    RawContentItem,
    RawMediaItem,
    RawMessage,
    RawTextItem,
    Role,
    StopReason,
    ToolPromptFormat,
)
from llama_stack.models.llama.llama3.chat_format import ChatFormat
from llama_stack.models.llama.llama3.prompt_templates import (
    BuiltinToolGenerator,
    FunctionTagCustomToolGenerator,
    JsonCustomToolGenerator,
    PythonListCustomToolGenerator,
    SystemDefaultGenerator,
)
from llama_stack.models.llama.llama3.tokenizer import Tokenizer
from llama_stack.models.llama.llama4.prompt_templates.system_prompts import (
    PythonListCustomToolGenerator as PythonListCustomToolGeneratorLlama4,
)
from llama_stack.models.llama.sku_list import resolve_model
from llama_stack.models.llama.sku_types import ModelFamily, is_multimodal
from llama_stack.providers.utils.inference import supported_inference_models

log = get_logger(name=__name__, category="providers::utils")


class ChatCompletionRequestWithRawContent(ChatCompletionRequest):
    messages: list[RawMessage]


class CompletionRequestWithRawContent(CompletionRequest):
    content: RawContent


def decode_assistant_message(content: str, stop_reason: StopReason) -> RawMessage:
    formatter = ChatFormat(Tokenizer.get_instance())
    return formatter.decode_assistant_message_from_content(content, stop_reason)


def interleaved_content_as_str(
    content: Any,
    sep: str = " ",
) -> str:
    if content is None:
        return ""

    def _process(c) -> str:
        if isinstance(c, str):
            return c
        elif isinstance(c, TextContentItem) or isinstance(c, OpenAIChatCompletionContentPartTextParam):
            return c.text
        elif isinstance(c, ImageContentItem) or isinstance(c, OpenAIChatCompletionContentPartImageParam):
            return "<image>"
        elif isinstance(c, OpenAIFile):
            return "<file>"
        else:
            raise ValueError(f"Unsupported content type: {type(c)}")

    if isinstance(content, list):
        return sep.join(_process(c) for c in content)
    else:
        return _process(content)


async def convert_request_to_raw(
    request: ChatCompletionRequest | CompletionRequest,
) -> ChatCompletionRequestWithRawContent | CompletionRequestWithRawContent:
    if isinstance(request, ChatCompletionRequest):
        messages = []
        for m in request.messages:
            content = await interleaved_content_convert_to_raw(m.content)
            d = m.model_dump()
            d["content"] = content
            messages.append(RawMessage(**d))

        d = request.model_dump()
        d["messages"] = messages
        request = ChatCompletionRequestWithRawContent(**d)
    else:
        d = request.model_dump()
        d["content"] = await interleaved_content_convert_to_raw(request.content)
        request = CompletionRequestWithRawContent(**d)

    return request


async def interleaved_content_convert_to_raw(
    content: InterleavedContent,
) -> RawContent:
    """Download content from URLs / files etc. so plain bytes can be sent to the model"""

    async def _localize_single(c: str | InterleavedContentItem) -> str | RawContentItem:
        if isinstance(c, str):
            return RawTextItem(text=c)
        elif isinstance(c, TextContentItem):
            return RawTextItem(text=c.text)
        elif isinstance(c, ImageContentItem):
            image = c.image
            if image.url:
                # Load image bytes from URL
                if image.url.uri.startswith("data"):
                    match = re.match(r"data:image/(\w+);base64,(.+)", image.url.uri)
                    if not match:
                        raise ValueError(f"Invalid data URL format, {image.url.uri[:40]}...")
                    _, image_data = match.groups()
                    data = base64.b64decode(image_data)
                elif image.url.uri.startswith("file://"):
                    path = image.url.uri[len("file://") :]
                    with open(path, "rb") as f:
                        data = f.read()  # type: ignore
                elif image.url.uri.startswith("http"):
                    async with httpx.AsyncClient() as client:
                        response = await client.get(image.url.uri)
                        data = response.content
                else:
                    raise ValueError("Unsupported URL type")
            elif image.data:
                # data is a base64 encoded string, decode it to bytes for RawMediaItem
                data = base64.b64decode(image.data)
            else:
                raise ValueError("No data or URL provided")

            return RawMediaItem(data=data)
        else:
            raise ValueError(f"Unsupported content type: {type(c)}")

    if isinstance(content, list):
        return await asyncio.gather(*(_localize_single(c) for c in content))
    else:
        return await _localize_single(content)


def content_has_media(content: InterleavedContent):
    def _has_media_content(c):
        return isinstance(c, ImageContentItem)

    if isinstance(content, list):
        return any(_has_media_content(c) for c in content)
    else:
        return _has_media_content(content)


def messages_have_media(messages: list[Message]):
    return any(content_has_media(m.content) for m in messages)


def request_has_media(request: ChatCompletionRequest | CompletionRequest):
    if isinstance(request, ChatCompletionRequest):
        return messages_have_media(request.messages)
    else:
        return content_has_media(request.content)


async def localize_image_content(uri: str) -> tuple[bytes, str] | None:
    if uri.startswith("http"):
        async with httpx.AsyncClient() as client:
            r = await client.get(uri)
            content = r.content
            content_type = r.headers.get("content-type")
            if content_type:
                format = content_type.split("/")[-1]
            else:
                format = "png"

        return content, format
    elif uri.startswith("data"):
        # data:image/{format};base64,{data}
        match = re.match(r"data:image/(\w+);base64,(.+)", uri)
        if not match:
            raise ValueError(f"Invalid data URL format, {uri[:40]}...")
        fmt, image_data = match.groups()
        content = base64.b64decode(image_data)
        return content, fmt
    else:
        return None


async def convert_image_content_to_url(
    media: ImageContentItem, download: bool = False, include_format: bool = True
) -> str:
    image = media.image
    if image.url and (not download or image.url.uri.startswith("data")):
        return image.url.uri

    if image.data:
        # data is a base64 encoded string, decode it to bytes first
        # TODO(mf): do this more efficiently, decode less
        content = base64.b64decode(image.data)
        pil_image = PIL_Image.open(io.BytesIO(content))
        format = pil_image.format
    else:
        localize_result = await localize_image_content(image.url.uri)
        if localize_result is None:
            raise ValueError(f"Failed to localize image content from {image.url.uri}")
        content, format = localize_result

    if include_format:
        return f"data:image/{format};base64," + base64.b64encode(content).decode("utf-8")
    else:
        return base64.b64encode(content).decode("utf-8")


def augment_content_with_response_format_prompt(response_format, content):
    if fmt_prompt := response_format_prompt(response_format):
        if isinstance(content, list):
            return content + [TextContentItem(text=fmt_prompt)]
        elif isinstance(content, str):
            return [TextContentItem(text=content), TextContentItem(text=fmt_prompt)]
        else:
            return [content, TextContentItem(text=fmt_prompt)]

    return content


async def chat_completion_request_to_prompt(request: ChatCompletionRequest, llama_model: str) -> str:
    messages = chat_completion_request_to_messages(request, llama_model)
    request.messages = messages
    request = await convert_request_to_raw(request)

    formatter = ChatFormat(tokenizer=Tokenizer.get_instance())
    model_input = formatter.encode_dialog_prompt(
        request.messages,
        tool_prompt_format=request.tool_config.tool_prompt_format or get_default_tool_prompt_format(llama_model),
    )
    return formatter.tokenizer.decode(model_input.tokens)


async def chat_completion_request_to_model_input_info(
    request: ChatCompletionRequest, llama_model: str
) -> tuple[str, int]:
    messages = chat_completion_request_to_messages(request, llama_model)
    request.messages = messages
    request = await convert_request_to_raw(request)

    formatter = ChatFormat(tokenizer=Tokenizer.get_instance())
    model_input = formatter.encode_dialog_prompt(
        request.messages,
        tool_prompt_format=request.tool_config.tool_prompt_format or get_default_tool_prompt_format(llama_model),
    )
    return (
        formatter.tokenizer.decode(model_input.tokens),
        len(model_input.tokens),
    )


def chat_completion_request_to_messages(
    request: ChatCompletionRequest,
    llama_model: str,
) -> list[Message]:
    """Reads chat completion request and augments the messages to handle tools.
    For eg. for llama_3_1, add system message with the appropriate tools or
    add user messsage for custom tools, etc.
    """
    assert llama_model is not None, "llama_model is required"
    model = resolve_model(llama_model)
    if model is None:
        log.error(f"Could not resolve model {llama_model}")
        return request.messages

    allowed_models = supported_inference_models()
    descriptors = [m.descriptor() for m in allowed_models]
    if model.descriptor() not in descriptors:
        log.error(f"Unsupported inference model? {model.descriptor()}")
        return request.messages

    if model.model_family == ModelFamily.llama3_1 or (
        model.model_family == ModelFamily.llama3_2 and is_multimodal(model.core_model_id)
    ):
        # llama3.1 and llama3.2 multimodal models follow the same tool prompt format
        messages = augment_messages_for_tools_llama_3_1(request)
    elif model.model_family in (
        ModelFamily.llama3_2,
        ModelFamily.llama3_3,
    ):
        # llama3.2, llama3.3 follow the same tool prompt format
        messages = augment_messages_for_tools_llama(request, PythonListCustomToolGenerator)
    elif model.model_family == ModelFamily.llama4:
        messages = augment_messages_for_tools_llama(request, PythonListCustomToolGeneratorLlama4)
    else:
        messages = request.messages

    if fmt_prompt := response_format_prompt(request.response_format):
        messages.append(UserMessage(content=fmt_prompt))

    return messages


def response_format_prompt(fmt: ResponseFormat | None):
    if not fmt:
        return None

    if fmt.type == ResponseFormatType.json_schema.value:
        return f"Please respond in JSON format with the schema: {json.dumps(fmt.json_schema)}"
    elif fmt.type == ResponseFormatType.grammar.value:
        raise NotImplementedError("Grammar response format not supported yet")
    else:
        raise ValueError(f"Unknown response format {fmt.type}")


def augment_messages_for_tools_llama_3_1(
    request: ChatCompletionRequest,
) -> list[Message]:
    existing_messages = request.messages
    existing_system_message = None
    if existing_messages[0].role == Role.system.value:
        existing_system_message = existing_messages.pop(0)

    assert existing_messages[0].role != Role.system.value, "Should only have 1 system message"

    messages = []

    default_gen = SystemDefaultGenerator()
    default_template = default_gen.gen()

    sys_content = ""

    tool_template = None
    if request.tools:
        tool_gen = BuiltinToolGenerator()
        tool_template = tool_gen.gen(request.tools)

        sys_content += tool_template.render()
        sys_content += "\n"

    sys_content += default_template.render()

    if existing_system_message:
        # TODO: this fn is needed in many places
        def _process(c):
            if isinstance(c, str):
                return c
            else:
                return "<media>"

        sys_content += "\n"

        if isinstance(existing_system_message.content, str):
            sys_content += _process(existing_system_message.content)
        elif isinstance(existing_system_message.content, list):
            sys_content += "\n".join([_process(c) for c in existing_system_message.content])

    tool_choice_prompt = _get_tool_choice_prompt(request.tool_config.tool_choice, request.tools)
    if tool_choice_prompt:
        sys_content += "\n" + tool_choice_prompt

    messages.append(SystemMessage(content=sys_content))

    has_custom_tools = request.tools is not None and any(isinstance(dfn.tool_name, str) for dfn in request.tools)
    if has_custom_tools:
        fmt = request.tool_config.tool_prompt_format or ToolPromptFormat.json
        if fmt == ToolPromptFormat.json:
            tool_gen = JsonCustomToolGenerator()
        elif fmt == ToolPromptFormat.function_tag:
            tool_gen = FunctionTagCustomToolGenerator()
        else:
            raise ValueError(f"Non supported ToolPromptFormat {fmt}")

        custom_tools = [t for t in request.tools if isinstance(t.tool_name, str)]
        custom_template = tool_gen.gen(custom_tools)
        messages.append(UserMessage(content=custom_template.render()))

    # Add back existing messages from the request
    messages += existing_messages

    return messages


def augment_messages_for_tools_llama(
    request: ChatCompletionRequest,
    custom_tool_prompt_generator,
) -> list[Message]:
    existing_messages = request.messages
    existing_system_message = None
    if existing_messages[0].role == Role.system.value:
        existing_system_message = existing_messages.pop(0)

    assert existing_messages[0].role != Role.system.value, "Should only have 1 system message"

    sys_content = ""
    custom_tools, builtin_tools = [], []
    for t in request.tools:
        if isinstance(t.tool_name, str):
            custom_tools.append(t)
        else:
            builtin_tools.append(t)

    if builtin_tools:
        tool_gen = BuiltinToolGenerator()
        tool_template = tool_gen.gen(builtin_tools)

        sys_content += tool_template.render()
        sys_content += "\n"

    custom_tools = [dfn for dfn in request.tools if isinstance(dfn.tool_name, str)]
    if custom_tools:
        fmt = request.tool_config.tool_prompt_format or ToolPromptFormat.python_list
        if fmt != ToolPromptFormat.python_list:
            raise ValueError(f"Non supported ToolPromptFormat {request.tool_config.tool_prompt_format}")

        system_prompt = None
        if existing_system_message and request.tool_config.system_message_behavior == SystemMessageBehavior.replace:
            system_prompt = existing_system_message.content

        tool_template = custom_tool_prompt_generator().gen(custom_tools, system_prompt)

        sys_content += tool_template.render()
        sys_content += "\n"

    if existing_system_message and (
        request.tool_config.system_message_behavior == SystemMessageBehavior.append or not custom_tools
    ):
        sys_content += interleaved_content_as_str(existing_system_message.content, sep="\n")

    tool_choice_prompt = _get_tool_choice_prompt(request.tool_config.tool_choice, request.tools)
    if tool_choice_prompt:
        sys_content += "\n" + tool_choice_prompt

    messages = [SystemMessage(content=sys_content.strip("\n")), *existing_messages]
    return messages


def _get_tool_choice_prompt(tool_choice: ToolChoice | str, tools: list[ToolDefinition]) -> str:
    if tool_choice == ToolChoice.auto:
        return ""
    elif tool_choice == ToolChoice.required:
        return "You MUST use one of the provided functions/tools to answer the user query."
    elif tool_choice == ToolChoice.none:
        # tools are already not passed in
        return ""
    else:
        # specific tool
        return f"You MUST use the tool `{tool_choice}` to answer the user query."


def get_default_tool_prompt_format(model: str) -> ToolPromptFormat:
    llama_model = resolve_model(model)
    if llama_model is None:
        log.warning(f"Could not resolve model {model}, defaulting to json tool prompt format")
        return ToolPromptFormat.json

    if llama_model.model_family == ModelFamily.llama3_1 or (
        llama_model.model_family == ModelFamily.llama3_2 and is_multimodal(llama_model.core_model_id)
    ):
        # llama3.1 and llama3.2 multimodal models follow the same tool prompt format
        return ToolPromptFormat.json
    elif llama_model.model_family in (
        ModelFamily.llama3_2,
        ModelFamily.llama3_3,
        ModelFamily.llama4,
    ):
        # llama3.2 and llama3.3 models follow the same tool prompt format
        return ToolPromptFormat.python_list
    else:
        return ToolPromptFormat.json
