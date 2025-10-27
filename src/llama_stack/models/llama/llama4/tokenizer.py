# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Collection, Iterator, Sequence, Set
from pathlib import Path
from typing import (
    Literal,
    cast,
)

import tiktoken

from llama_stack.log import get_logger
from llama_stack.models.llama.tokenizer_utils import load_bpe_file

# The tiktoken tokenizer can handle <=400k chars without
# pyo3_runtime.PanicException.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# https://github.com/openai/tiktoken/issues/195
# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000


_INSTANCE = None


def get_reserved_special_tokens(name, count, start_index=0):
    return [f"<|{name}_reserved_special_token_{i}|>" for i in range(start_index, start_index + count)]


# 200005, ..., 200079
LLAMA4_TEXT_POST_TRAIN_SPECIAL_TOKENS = [
    "<|header_start|>",
    "<|header_end|>",
    "<|eom|>",
    "<|eot|>",
    "<|step|>",
    "<|text_post_train_reserved_special_token_0|>",
    "<|text_post_train_reserved_special_token_1|>",
    "<|text_post_train_reserved_special_token_2|>",
    "<|text_post_train_reserved_special_token_3|>",
    "<|text_post_train_reserved_special_token_4|>",
    "<|text_post_train_reserved_special_token_5|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|finetune_right_pad|>",
] + get_reserved_special_tokens(
    "text_post_train", 61, 8
)  # <|text_post_train_reserved_special_token_6|>, ..., <|text_post_train_reserved_special_token_66|>

# 200080, ..., 201133
LLAMA4_VISION_SPECIAL_TOKENS = [
    "<|image_start|>",
    "<|image_end|>",
    "<|vision_reserved_special_token_0|>",
    "<|vision_reserved_special_token_1|>",
    "<|tile_x_separator|>",
    "<|tile_y_separator|>",
    "<|vision_reserved_special_token_2|>",
    "<|vision_reserved_special_token_3|>",
    "<|vision_reserved_special_token_4|>",
    "<|vision_reserved_special_token_5|>",
    "<|image|>",
    "<|vision_reserved_special_token_6|>",
    "<|patch|>",
] + get_reserved_special_tokens(
    "vision", 1041, 7
)  # <|vision_reserved_special_token_7|>, ..., <|vision_reserved_special_token_1047|>

# 201134, ..., 201143
LLAMA4_REASONING_SPECIAL_TOKENS = [
    "<|reasoning_reserved_special_token_0|>",
    "<|reasoning_reserved_special_token_1|>",
    "<|reasoning_reserved_special_token_2|>",
    "<|reasoning_reserved_special_token_3|>",
    "<|reasoning_reserved_special_token_4|>",
    "<|reasoning_reserved_special_token_5|>",
    "<|reasoning_reserved_special_token_6|>",
    "<|reasoning_reserved_special_token_7|>",
    "<|reasoning_thinking_start|>",
    "<|reasoning_thinking_end|>",
]

LLAMA4_SPECIAL_TOKENS = (
    LLAMA4_TEXT_POST_TRAIN_SPECIAL_TOKENS + LLAMA4_VISION_SPECIAL_TOKENS + LLAMA4_REASONING_SPECIAL_TOKENS
)

BASIC_SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
]

logger = get_logger(name=__name__, category="models::llama")


class Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: dict[str, int]

    num_reserved_special_tokens = 2048

    O200K_PATTERN = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # noqa: E501

    @classmethod
    def get_instance(cls):
        global _INSTANCE

        if _INSTANCE is None:
            _INSTANCE = Tokenizer(Path(__file__).parent / "tokenizer.model")
        return _INSTANCE

    def __init__(self, model_path: Path):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (Path): The path to the Tiktoken model file.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Tokenizer model file not found: {model_path}")

        mergeable_ranks = load_bpe_file(model_path)
        num_base_tokens = len(mergeable_ranks)

        special_tokens = BASIC_SPECIAL_TOKENS + LLAMA4_SPECIAL_TOKENS
        assert len(set(special_tokens)) == len(special_tokens)
        assert len(special_tokens) <= self.num_reserved_special_tokens

        reserved_tokens = [
            f"<|reserved_special_token_{i}|>" for i in range(self.num_reserved_special_tokens - len(special_tokens))
        ]
        special_tokens = special_tokens + reserved_tokens

        self.special_tokens = {token: num_base_tokens + i for i, token in enumerate(special_tokens)}
        self.model = tiktoken.Encoding(
            name=model_path.name,
            pat_str=self.O200K_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.n_words: int = num_base_tokens + len(special_tokens)

        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]

        self.pad_id: int = self.special_tokens["<|finetune_right_pad|>"]
        self.eot_id: int = self.special_tokens["<|eot|>"]
        self.eom_id: int = self.special_tokens["<|eom|>"]

        self.thinking_start_id: int = self.special_tokens["<|reasoning_thinking_start|>"]
        self.thinking_end_id: int = self.special_tokens["<|reasoning_thinking_end|>"]

        self.stop_tokens = [
            self.eos_id,
            self.special_tokens["<|eom|>"],
            self.special_tokens["<|eot|>"],
        ]

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Literal["all"] | Set[str] | None = None,
        disallowed_special: Literal["all"] | Collection[str] = (),
    ) -> list[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_special ("all"|set[str]): allowed special tokens in string
            disallowed_special ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        if allowed_special is None:
            allowed_special = set()
        assert type(s) is str

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: list[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(list[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]
