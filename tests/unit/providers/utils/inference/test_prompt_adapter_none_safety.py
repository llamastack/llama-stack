# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for None-safety in interleaved_content_as_str.

Regression tests for https://github.com/meta-llama/llama-stack/issues/4996
Models like Qwen3 via vLLM may return tool-call messages with content: null,
causing str.join() to crash with TypeError when None values flow through.
"""

from llama_stack.providers.utils.inference.prompt_adapter import interleaved_content_as_str


class TestInterleavedContentAsStrNoneSafety:
    """Ensure interleaved_content_as_str never passes None into join()."""

    def test_none_content_returns_empty_string(self):
        assert interleaved_content_as_str(None) == ""

    def test_string_content(self):
        assert interleaved_content_as_str("hello") == "hello"

    def test_list_of_strings(self):
        assert interleaved_content_as_str(["a", "b", "c"]) == "a b c"

    def test_empty_list(self):
        assert interleaved_content_as_str([]) == ""

    def test_text_content_item_with_none_text(self):
        """TextContentItem with text=None must not crash join().

        When an upstream provider (e.g. vLLM) returns content: null,
        the deserialized object may have text=None despite the Pydantic
        schema requiring str.  model_construct() simulates this.
        """
        from llama_stack_api.common.content_types import TextContentItem

        item = TextContentItem.model_construct(text=None)
        result = interleaved_content_as_str([item])
        assert result == ""

    def test_text_content_item_with_valid_text(self):
        from llama_stack_api.common.content_types import TextContentItem

        item = TextContentItem(text="hello world")
        result = interleaved_content_as_str([item])
        assert result == "hello world"
