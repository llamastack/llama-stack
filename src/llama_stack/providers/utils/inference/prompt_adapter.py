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
    CompletionRequest,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIFile,
    OpenAIMessageParam,
    OpenAISystemMessageParam,
    OpenAIToolMessageParam,
    OpenAIUserMessageParam,
    ResponseFormat,
    ResponseFormatType,
    ToolChoice,
)
from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import (
    RawContent,
    RawContentItem,
    RawMediaItem,
    RawMessage,
    RawTextItem,
    StopReason,
    ToolCall,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.models.llama.llama3.chat_format import ChatFormat
from llama_stack.models.llama.llama3.tokenizer import Tokenizer
from llama_stack.models.llama.sku_list import resolve_model
from llama_stack.models.llama.sku_types import ModelFamily, is_multimodal

log = get_logger(name=__name__, category="providers::utils")


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


async def convert_openai_message_to_raw_message(message: OpenAIMessageParam) -> RawMessage:
    """Convert OpenAI message format to RawMessage format used by Llama formatters."""
    if isinstance(message, OpenAIUserMessageParam):
        content = await interleaved_content_convert_to_raw(message.content)  # type: ignore[arg-type]
        return RawMessage(role="user", content=content)
    elif isinstance(message, OpenAISystemMessageParam):
        content = await interleaved_content_convert_to_raw(message.content)  # type: ignore[arg-type]
        return RawMessage(role="system", content=content)
    elif isinstance(message, OpenAIAssistantMessageParam):
        content = await interleaved_content_convert_to_raw(message.content or "")  # type: ignore[arg-type]
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc.function:
                    tool_calls.append(
                        ToolCall(
                            call_id=tc.id or "",
                            tool_name=tc.function.name or "",
                            arguments=tc.function.arguments or "{}",
                        )
                    )
        return RawMessage(role="assistant", content=content, tool_calls=tool_calls)
    elif isinstance(message, OpenAIToolMessageParam):
        content = await interleaved_content_convert_to_raw(message.content)  # type: ignore[arg-type]
        return RawMessage(role="tool", content=content)
    else:
        # Handle OpenAIDeveloperMessageParam if needed
        raise ValueError(f"Unsupported message type: {type(message)}")


def content_has_media(content: InterleavedContent):
    def _has_media_content(c):
        return isinstance(c, ImageContentItem)

    if isinstance(content, list):
        return any(_has_media_content(c) for c in content)
    else:
        return _has_media_content(content)


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


def response_format_prompt(fmt: ResponseFormat | None):
    if not fmt:
        return None

    if fmt.type == ResponseFormatType.json_schema.value:
        return f"Please respond in JSON format with the schema: {json.dumps(fmt.json_schema)}"
    elif fmt.type == ResponseFormatType.grammar.value:
        raise NotImplementedError("Grammar response format not supported yet")
    else:
        raise ValueError(f"Unknown response format {fmt.type}")


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
