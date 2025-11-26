# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import json
import uuid
from dataclasses import dataclass

import torch
from PIL import Image as PIL_Image

# TODO: either fork these or move them to the common package
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
from ..llama3.tool_utils import ToolUtils
from .args import VisionArgs
from .datatypes import LLMInput
from .preprocess import ResizeNormalizeImageTransform, VariableSizeImageTransform
from .tokenizer import Tokenizer


def role_str(role: Role) -> str:
    role_strs = {
        Role.user: "user",
        Role.system: "system",
        Role.tool: "ipython",  # special
        Role.assistant: "assistant",
    }
    return role_strs[role]


@dataclass
class TransformedImage:
    image_tiles: torch.Tensor
    # is the aspect ratio needed anywhere?
    aspect_ratio: tuple[int, int]


def convert_image_to_rgb(image: PIL_Image.Image, bg: tuple[int, int, int] = (255, 255, 255)) -> PIL_Image.Image:
    if image.mode == "RGBA":
        image.load()  # for png.split()
        new_img = PIL_Image.new("RGB", image.size, bg)
        new_img.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return new_img
    return image.convert("RGB")


class ChatFormat:
    possible_headers: dict[Role, str]

    def __init__(
        self,
        tokenizer: Tokenizer,
        vision_args: VisionArgs | None = None,
        max_num_chunks: int = 16,
    ):
        self.tokenizer = tokenizer
        self.vision_args = vision_args
        self.max_num_chunks = max_num_chunks

        self.possible_headers = {role: f"<|header_start|>{role_str(role)}<|header_end|>\n\n" for role in Role}

        self.image_transform = None
        self.dynamic_image_transform = None
        if vision_args:
            self.dynamic_image_transform = VariableSizeImageTransform(vision_args.image_size.width)
            self.image_transform = ResizeNormalizeImageTransform(
                vision_args.image_size.width, vision_args.image_size.height
            )

    def _encode_header(self, role: str) -> list[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|header_start|>"])

        # TODO: need to check if this is correct
        tokens.extend(self.tokenizer.encode("ipython" if role == "tool" else role, bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|header_end|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_content(self, content: RawContent) -> LLMInput:
        tokens, images = self._encode_content(content, bos=True)
        return self._model_input_from_tokens_images(tokens, images)

    def _encode_image(
        self,
        transformed_image: TransformedImage,
    ) -> list[int]:
        assert self.vision_args is not None, "The model is not vision-enabled"

        image_tensor = transformed_image.image_tiles
        image_channels = image_tensor.shape[-3]
        image_height = image_tensor.shape[-2]
        image_width = image_tensor.shape[-1]
        image_chunks = image_tensor.view(-1, image_channels, image_height, image_width).shape[0]

        patch_height = self.vision_args.patch_size.height
        patch_width = self.vision_args.patch_size.width

        if image_height % patch_height != 0:
            raise ValueError(f"{image_height=} not divisible by {patch_height=}")
        if image_width % patch_width != 0:
            raise ValueError(f"{image_width=} not divisible by {patch_width=}")

        ds_ratio = int(round(1.0 / (self.vision_args.pixel_shuffle_ratio**2)))
        n_patches_per_chunk = int((image_height // patch_height) * (image_width // patch_width) // ds_ratio)

        image_ar = transformed_image.aspect_ratio
        tokens = [self.tokenizer.special_tokens["<|image_start|>"]]
        if image_chunks == 1:
            tokens += [self.tokenizer.special_tokens["<|image|>"]]
            tokens += [self.tokenizer.special_tokens["<|patch|>"]] * n_patches_per_chunk
            tokens += [self.tokenizer.special_tokens["<|image_end|>"]]
        else:
            ratio_h, ratio_w = image_ar
            for _ in range(ratio_h):
                for xx in range(ratio_w):
                    tokens += [self.tokenizer.special_tokens["<|patch|>"]] * n_patches_per_chunk
                    if xx < ratio_w - 1:
                        tokens.append(self.tokenizer.special_tokens["<|tile_x_separator|>"])

                tokens.append(self.tokenizer.special_tokens["<|tile_y_separator|>"])

            tokens += [self.tokenizer.special_tokens["<|image|>"]]
            tokens += [self.tokenizer.special_tokens["<|patch|>"]] * n_patches_per_chunk
            tokens += [self.tokenizer.special_tokens["<|image_end|>"]]

        return tokens

    def _encode_content(self, content: RawContent, bos: bool = False) -> tuple[list[int], list[TransformedImage]]:
        tokens = []
        tranformed_images = []

        added_bos = False

        def _process(c):
            nonlocal added_bos, bos

            if isinstance(c, str) or isinstance(c, RawTextItem):
                if isinstance(c, RawTextItem):
                    c = c.text
                tokens.extend(self.tokenizer.encode(c, bos=False if added_bos else bos, eos=False))
                added_bos = True

            elif isinstance(c, RawMediaItem):
                if not self.vision_args:
                    raise ValueError("The model is not vision-enabled, but a media item was found")

                bos = False if added_bos else bos
                if bos:
                    tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
                    added_bos = True

                bytes_io = io.BytesIO(c.data) if isinstance(c.data, bytes) else c.data
                image = PIL_Image.open(bytes_io)
                image = convert_image_to_rgb(image)
                image_tiles, ar = self.dynamic_image_transform(image, max_num_chunks=self.max_num_chunks)

                if image_tiles.shape[0] > 1:
                    image_global = self.image_transform(image)
                    image_global = image_global.unsqueeze(0)
                    image_combine = torch.cat((image_tiles, image_global), dim=0)
                    image_tiles = image_combine

                transformed_image = TransformedImage(image_tiles=image_tiles, aspect_ratio=ar)
                tokens.extend(self._encode_image(transformed_image))
                tranformed_images.append(transformed_image)

        if isinstance(content, list):
            for c in content:
                _process(c)
        else:
            _process(content)

        return tokens, tranformed_images

    def encode_message(
        self, message: RawMessage, tool_prompt_format: ToolPromptFormat
    ) -> tuple[list[int], list[TransformedImage]]:
        tokens = self._encode_header(message.role)
        images = []

        def _process_content(c):
            toks, imgs = self._encode_content(c)
            tokens.extend(toks)
            images.extend(imgs)

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

        # Tool calls and Tool Response messages should be eom
        eom = False
        if message.role == "assistant":
            eom = message.stop_reason == StopReason.end_of_message or message.tool_calls
        elif message.role == "tool":
            eom = True

        tokens.append(self.tokenizer.special_tokens["<|eom|>" if eom else "<|eot|>"])
        return tokens, images

    def encode_dialog_prompt(
        self,
        messages: list[RawMessage],
        tool_prompt_format: ToolPromptFormat = ToolPromptFormat.json,
    ) -> LLMInput:
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

        ipython = content.startswith("<|python_start|>")
        if ipython:
            content = content[len("<|python_start|>") :]
            content = content.replace("<|python_end|>", "")

        if content.endswith("<|eot|>"):
            content = content[: -len("<|eot|>")]
            stop_reason = StopReason.end_of_turn
        elif content.endswith("<|eom|>"):
            content = content[: -len("<|eom|>")]
            stop_reason = StopReason.end_of_message

        tool_name = None
        tool_arguments = {}

        custom_tool_info = ToolUtils.maybe_extract_custom_tool_call(content)
        if custom_tool_info is not None:
            tool_name, tool_arguments = custom_tool_info
            # Sometimes when agent has custom tools alongside builin tools
            # Agent responds for builtin tool calls in the format of the custom tools
            # This code tries to handle that case
            if tool_name in BuiltinTool.__members__:
                tool_name = BuiltinTool[tool_name]
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

    def _model_input_from_tokens_images(self, tokens: list[int], images: list[TransformedImage]) -> LLMInput:
        return LLMInput(
            tokens=tokens,
            images=[x.image_tiles for x in images] if len(images) > 0 else None,
        )
