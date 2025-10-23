# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import json
import uuid
from dataclasses import dataclass
from typing import Any

from PIL import Image as PIL_Image

from ..datatypes import (
    BuiltinTool,
    RawContent,
    RawMediaItem,
    RawMessage,
    RawTextItem,
    Role,
    StopReason,
    ToolCall,
    ToolPromptFormat,
)
from .tokenizer import Tokenizer
from .tool_utils import ToolUtils


@dataclass
class VisionInput:
    mask: list[list[int]]
    images: list[PIL_Image.Image]


@dataclass
class LLMInput:
    tokens: list[int]
    vision: VisionInput | None = None


def role_str(role: Role) -> str:
    role_strs = {
        Role.user: "user",
        Role.system: "system",
        Role.tool: "ipython",  # special
        Role.assistant: "assistant",
    }
    return role_strs[role]


class ChatFormat:
    possible_headers: dict[Role, str]

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

        self.possible_headers = {role: f"<|start_header_id|>{role_str(role)}<|end_header_id|>\n\n" for role in Role}
        self.vision_token = self.tokenizer.special_tokens["<|image|>"]

    def _encode_header(self, role: str) -> list[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode("ipython" if role == "tool" else role, bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_content(self, content: RawContent) -> LLMInput:
        tokens, images = self._encode_content(content, bos=True)
        return self._model_input_from_tokens_images(tokens, images)

    def _encode_content(self, content: RawContent, bos: bool = False) -> tuple[list[int], list[PIL_Image.Image]]:
        tokens = []
        images = []

        added_bos = False

        def _process(c):
            nonlocal added_bos, bos

            if isinstance(c, str) or isinstance(c, RawTextItem):
                if isinstance(c, RawTextItem):
                    c = c.text
                tokens.extend(self.tokenizer.encode(c, bos=False if added_bos else bos, eos=False))
                added_bos = True

            elif isinstance(c, RawMediaItem):
                bos = False if added_bos else bos
                if bos:
                    tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
                    added_bos = True
                tokens.append(self.vision_token)

                bytes_io = io.BytesIO(c.data) if isinstance(c.data, bytes) else c.data
                image = PIL_Image.open(bytes_io)
                image = image.convert("RGB")
                images.append(image)

        if isinstance(content, list):
            for c in content:
                _process(c)
        else:
            _process(content)

        return tokens, images

    def encode_message(
        self, message: RawMessage, tool_prompt_format: ToolPromptFormat
    ) -> tuple[list[int], list[PIL_Image.Image]]:
        tokens = self._encode_header(message.role)
        images = []

        def _process_content(c):
            toks, imgs = self._encode_content(c)
            tokens.extend(toks)
            images.extend(imgs)

        if (
            message.role == "assistant"
            and len(message.tool_calls) > 0
            and message.tool_calls[0].tool_name == BuiltinTool.code_interpreter
        ):
            tokens.append(self.tokenizer.special_tokens["<|python_tag|>"])

        _process_content(message.content)

        if message.role == "user" and message.context is not None:
            # This is RAG context; why is it here in the chat format? I don't think
            # this is needed and can be moved upwards
            _process_content("\n\n")
            _process_content(message.context)

        if message.role == "assistant":
            for t in message.tool_calls:
                content = ToolUtils.encode_tool_call(t, tool_prompt_format)
                _process_content(content)

        eom = False
        if message.role == "assistant":
            eom = message.stop_reason == StopReason.end_of_message

        tokens.append(self.tokenizer.special_tokens["<|eom_id|>" if eom else "<|eot_id|>"])
        return tokens, images

    def encode_dialog_prompt(
        self,
        messages: list[RawMessage],
        tool_prompt_format: ToolPromptFormat | None = None,
    ) -> LLMInput:
        tool_prompt_format = tool_prompt_format or ToolPromptFormat.json
        tokens = []
        images = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        for message in messages:
            toks, imgs = self.encode_message(message, tool_prompt_format)
            tokens.extend(toks)
            images.extend(imgs)

        # Add the start of an assistant message for the model to complete.
        tokens.extend(self._encode_header("assistant"))

        return self._model_input_from_tokens_images(tokens, images)

    # TODO(this should be generic, not only for assistant messages)
    def decode_assistant_message(self, tokens: list[int], stop_reason: StopReason) -> RawMessage:
        content = self.tokenizer.decode(tokens)

        return self.decode_assistant_message_from_content(content, stop_reason)

    def decode_assistant_message_from_content(self, content: str, stop_reason: StopReason) -> RawMessage:
        content = content.strip(" ")
        header_str = self.possible_headers[Role.assistant]
        if content.startswith(header_str):
            content = content[len(header_str) :]

        ipython = content.startswith("<|python_tag|>")
        if ipython:
            content = content[len("<|python_tag|>") :]

        if content.endswith("<|eot_id|>"):
            content = content[: -len("<|eot_id|>")]
            stop_reason = StopReason.end_of_turn
        elif content.endswith("<|eom_id|>"):
            content = content[: -len("<|eom_id|>")]
            stop_reason = StopReason.end_of_message

        tool_name: str | BuiltinTool | None = None
        tool_arguments: dict[str, Any] = {}

        custom_tool_info = ToolUtils.maybe_extract_custom_tool_call(content)
        if custom_tool_info is not None:
            # Type guard: ensure custom_tool_info is a tuple of correct types
            if isinstance(custom_tool_info, tuple) and len(custom_tool_info) == 2:
                extracted_tool_name, extracted_tool_arguments = custom_tool_info
                # Handle both dict and str return types from the function
                if isinstance(extracted_tool_arguments, dict):
                    tool_name, tool_arguments = extracted_tool_name, extracted_tool_arguments
                else:
                    # If it's a string, treat it as a query parameter
                    tool_name, tool_arguments = extracted_tool_name, {"query": extracted_tool_arguments}
            else:
                tool_name, tool_arguments = None, {}
            # Sometimes when agent has custom tools alongside builin tools
            # Agent responds for builtin tool calls in the format of the custom tools
            # This code tries to handle that case
            if tool_name is not None and tool_name in BuiltinTool.__members__:
                tool_name = BuiltinTool[tool_name]
                if isinstance(tool_arguments, dict):
                    tool_arguments = {
                        "query": list(tool_arguments.values())[0],
                    }
        else:
            builtin_tool_info = ToolUtils.maybe_extract_builtin_tool_call(content)
            if builtin_tool_info is not None:
                tool_name, query = builtin_tool_info
                tool_arguments = {
                    "query": query,
                }
                if tool_name in BuiltinTool.__members__:
                    tool_name = BuiltinTool[tool_name]
            elif ipython:
                tool_name = BuiltinTool.code_interpreter
                tool_arguments = {
                    "code": content,
                }

        tool_calls = []
        if tool_name is not None and tool_arguments is not None:
            call_id = str(uuid.uuid4())
            tool_calls.append(
                ToolCall(
                    call_id=call_id,
                    tool_name=tool_name,
                    arguments=json.dumps(tool_arguments),
                )
            )
            content = ""

        return RawMessage(
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
        )

    def _model_input_from_tokens_images(self, tokens: list[int], images: list[PIL_Image.Image]) -> LLMInput:
        vision_input = None
        if len(images) > 0:
            vision_input = VisionInput(
                mask=create_vision_mask(tokens, self.vision_token),
                images=images,
            )

        return LLMInput(
            tokens=[128256 if token == self.vision_token else token for token in tokens],
            vision=vision_input,
        )


def create_vision_mask(
    tokens: list[int],
    vision_token: int,
) -> list[list[int]]:
    vision_token_locations = [i for i, token in enumerate(tokens) if token == vision_token]
    if len(vision_token_locations) == 0:
        return []

    if len(vision_token_locations) == 1:
        # only one image present, unmask until end of sequence
        return [[vision_token_locations[0], -1]]
    vision_masks = [
        [loc1, loc2] for loc1, loc2 in zip(vision_token_locations[:-1], vision_token_locations[1:], strict=False)
    ]
    # last image will attend to all subsequent text
    vision_masks.append([vision_token_locations[-1], len(tokens)])

    # if there are two or more consecutive vision tokens,
    # they should all attend to all subsequent
    # text present
    last_mask_end = vision_masks[-1][1]
    for vision_mask in vision_masks[::-1]:
        if vision_mask[0] == vision_mask[1] - 1:
            vision_mask[1] = last_mask_end
        last_mask_end = vision_mask[1]
    return vision_masks
