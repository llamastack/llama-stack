# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import re
from typing import Any

import httpx

from llama_stack.log import get_logger
from llama_stack_api import (
    ImageContentItem,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIFile,
    TextContentItem,
)

log = get_logger(name=__name__, category="providers::utils")


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


_image_cache: dict[str, tuple[bytes, str]] = {}


async def localize_image_content(uri: str) -> tuple[bytes, str] | None:
    if uri.startswith("http"):
        cached = _image_cache.get(uri)
        if cached is not None:
            return cached

        async with httpx.AsyncClient() as client:
            r = await client.get(uri)
            content = r.content
            content_type = r.headers.get("content-type")
            if content_type:
                format = content_type.split("/")[-1]
            else:
                format = "png"

        result = (content, format)
        _image_cache[uri] = result
        return result
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
