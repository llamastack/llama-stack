# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Regression tests for None-safe prompt content joins."""

from llama_stack.providers.utils.inference.prompt_adapter import interleaved_content_as_str
from llama_stack_api.common.content_types import TextContentItem


class TestInterleavedContentAsStrNoneSafety:
    def test_none_content_returns_empty_string(self):
        assert interleaved_content_as_str(None) == ""

    def test_text_content_item_with_none_text_returns_empty_string(self):
        item = TextContentItem.model_construct(text=None)

        assert interleaved_content_as_str([item]) == ""

    def test_text_content_item_with_valid_text_is_preserved(self):
        item = TextContentItem(text="hello world")

        assert interleaved_content_as_str([item]) == "hello world"
