# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack_api import validate_embedding_input_is_text


class TestEmbeddingValidation:
    """Test the validate_embedding_input_is_text function."""

    def test_valid_string_input(self):
        """Test that string input is accepted."""
        # Should not raise
        validate_embedding_input_is_text("hello world", "test-provider")

    def test_valid_list_of_strings_input(self):
        """Test that list of strings is accepted."""
        # Should not raise
        validate_embedding_input_is_text(["hello", "world"], "test-provider")

    def test_invalid_list_of_ints_input(self):
        """Test that list of ints (token array) is rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_embedding_input_is_text([1, 2, 3], "test-provider")

        error_msg = str(exc_info.value)
        assert "test-provider" in error_msg
        assert "does not support token arrays" in error_msg

    def test_invalid_list_of_list_of_ints_input(self):
        """Test that list of list of ints (batch token array) is rejected."""
        with pytest.raises(ValueError) as exc_info:
            validate_embedding_input_is_text([[1, 2, 3], [4, 5, 6]], "test-provider")

        error_msg = str(exc_info.value)
        assert "test-provider" in error_msg
        assert "does not support token arrays" in error_msg

    def test_error_message_includes_provider_name(self):
        """Test that error message includes the provider name."""
        provider_names = ["litellm", "sentence-transformers", "together", "gemini", "watsonx"]

        for provider in provider_names:
            with pytest.raises(ValueError) as exc_info:
                validate_embedding_input_is_text([1, 2, 3], provider)

            error_msg = str(exc_info.value)
            assert provider in error_msg
