# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for tokenization utilities."""

import pytest

from llama_stack.providers.utils.inference.tokenization import (
    TokenizationError,
    clear_tokenizer_cache,
    count_tokens,
    get_tokenization_method,
)


class TestCountTokens:
    """Test suite for count_tokens function."""

    def test_count_tokens_simple_text_openai(self):
        """Test token counting for simple text with OpenAI models."""
        message = {"role": "user", "content": "Hello, world!"}

        # Should work with GPT-4
        token_count = count_tokens(message, model="gpt-4")
        assert isinstance(token_count, int)
        assert token_count > 0
        # "Hello, world!" should be around 3-4 tokens
        assert 2 <= token_count <= 5

    def test_count_tokens_simple_text_gpt4o(self):
        """Test token counting for GPT-4o model."""
        message = {"role": "user", "content": "This is a test message."}

        token_count = count_tokens(message, model="gpt-4o")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_empty_message(self):
        """Test token counting for empty message."""
        message = {"role": "user", "content": ""}

        token_count = count_tokens(message, model="gpt-4")
        assert token_count == 0

    def test_count_tokens_none_content(self):
        """Test token counting for None content."""
        message = {"role": "user", "content": None}

        token_count = count_tokens(message, model="gpt-4")
        assert token_count == 0

    def test_count_tokens_multiple_messages(self):
        """Test token counting for multiple messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather?"},
        ]

        token_count = count_tokens(messages, model="gpt-4")
        assert isinstance(token_count, int)
        assert token_count > 0
        # Should be more than single message
        assert token_count >= 10

    def test_count_tokens_long_text(self):
        """Test token counting for long text."""
        long_text = " ".join(["word"] * 1000)
        message = {"role": "user", "content": long_text}

        token_count = count_tokens(message, model="gpt-4")
        assert isinstance(token_count, int)
        # 1000 words should be close to 1000 tokens
        assert 900 <= token_count <= 1100

    def test_count_tokens_multimodal_text_only(self):
        """Test token counting for multimodal message with text only."""
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
            ],
        }

        token_count = count_tokens(message, model="gpt-4o")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_multimodal_with_image_low_res(self):
        """Test token counting for multimodal message with low-res image."""
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                        "detail": "low",
                    },
                },
            ],
        }

        token_count = count_tokens(message, model="gpt-4o")
        assert isinstance(token_count, int)
        # Should include text tokens + image tokens (85 for low-res)
        assert token_count >= 85

    def test_count_tokens_multimodal_with_image_high_res(self):
        """Test token counting for multimodal message with high-res image."""
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                        "detail": "high",
                    },
                },
            ],
        }

        token_count = count_tokens(message, model="gpt-4o")
        assert isinstance(token_count, int)
        # Should include text tokens + image tokens (170 for high-res)
        assert token_count >= 170

    def test_count_tokens_multimodal_with_image_auto(self):
        """Test token counting for multimodal message with auto detail."""
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"},
                },
            ],
        }

        token_count = count_tokens(message, model="gpt-4o")
        assert isinstance(token_count, int)
        # Should use average of low and high
        assert token_count >= 100

    def test_count_tokens_multiple_images(self):
        """Test token counting for message with multiple images."""
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these images."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image1.jpg",
                        "detail": "low",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image2.jpg",
                        "detail": "low",
                    },
                },
            ],
        }

        token_count = count_tokens(message, model="gpt-4o")
        assert isinstance(token_count, int)
        # Should include text + 2 * 85 tokens for images
        assert token_count >= 170

    def test_count_tokens_unknown_model_estimation(self):
        """Test token counting falls back to estimation for unknown models."""
        message = {"role": "user", "content": "Hello, world!"}

        # Unknown model should use character-based estimation
        token_count = count_tokens(message, model="unknown-model-xyz")
        assert isinstance(token_count, int)
        assert token_count > 0
        # "Hello, world!" is 13 chars, should estimate ~3-4 tokens
        assert 2 <= token_count <= 5

    def test_count_tokens_llama_model_fallback(self):
        """Test token counting for Llama models (may fall back to estimation)."""
        message = {"role": "user", "content": "Hello from Llama!"}

        # This may fail if transformers/model not available, should fall back
        token_count = count_tokens(
            message,
            model="meta-llama/Llama-3.1-8B-Instruct",
        )
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_with_exact_false(self):
        """Test token counting with exact=False uses estimation."""
        message = {"role": "user", "content": "This is a test."}

        token_count = count_tokens(message, model="gpt-4", exact=False)
        assert isinstance(token_count, int)
        assert token_count > 0
        # Should use character-based estimation
        # "This is a test." is 15 chars, should estimate ~3-4 tokens
        assert 3 <= token_count <= 5

    def test_count_tokens_malformed_message(self):
        """Test token counting with malformed message."""
        # Not a dict
        token_count = count_tokens("not a message", model="gpt-4")  # type: ignore
        assert token_count == 0

        # Missing content
        token_count = count_tokens({"role": "user"}, model="gpt-4")
        assert token_count == 0

        # Malformed content list
        message = {
            "role": "user",
            "content": [
                "not a dict",  # Invalid item
                {"type": "text", "text": "valid text"},
            ],
        }
        token_count = count_tokens(message, model="gpt-4")
        # Should only count valid items
        assert token_count > 0

    def test_count_tokens_empty_list(self):
        """Test token counting with empty message list."""
        token_count = count_tokens([], model="gpt-4")
        assert token_count == 0

    def test_count_tokens_special_characters(self):
        """Test token counting with special characters."""
        message = {"role": "user", "content": "Hello! @#$%^&*() 🎉"}

        token_count = count_tokens(message, model="gpt-4")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_very_long_text(self):
        """Test token counting with very long text (>1024 tokens)."""
        # Create text that should be >1024 tokens
        long_text = " ".join(["word"] * 2000)
        message = {"role": "user", "content": long_text}

        token_count = count_tokens(message, model="gpt-4")
        assert isinstance(token_count, int)
        # Should be close to 2000 tokens
        assert token_count >= 1024  # At least cacheable threshold
        assert 1800 <= token_count <= 2200

    def test_count_tokens_fine_tuned_model(self):
        """Test token counting for fine-tuned OpenAI model."""
        message = {"role": "user", "content": "Test fine-tuned model."}

        # Fine-tuned models should still work
        token_count = count_tokens(message, model="gpt-4-turbo-2024-04-09")
        assert isinstance(token_count, int)
        assert token_count > 0


class TestGetTokenizationMethod:
    """Test suite for get_tokenization_method function."""

    def test_get_tokenization_method_openai(self):
        """Test getting tokenization method for OpenAI models."""
        assert get_tokenization_method("gpt-4") == "exact-tiktoken"
        assert get_tokenization_method("gpt-4o") == "exact-tiktoken"
        assert get_tokenization_method("gpt-3.5-turbo") == "exact-tiktoken"
        assert get_tokenization_method("gpt-4-turbo") == "exact-tiktoken"

    def test_get_tokenization_method_llama(self):
        """Test getting tokenization method for Llama models."""
        assert (
            get_tokenization_method("meta-llama/Llama-3.1-8B-Instruct")
            == "exact-transformers"
        )
        assert (
            get_tokenization_method("meta-llama/Llama-4-Scout-17B-16E-Instruct")
            == "exact-transformers"
        )
        assert (
            get_tokenization_method("meta-llama/Meta-Llama-3-8B")
            == "exact-transformers"
        )

    def test_get_tokenization_method_unknown(self):
        """Test getting tokenization method for unknown models."""
        assert get_tokenization_method("unknown-model") == "estimated"
        assert get_tokenization_method("claude-3") == "estimated"
        assert get_tokenization_method("random-model-xyz") == "estimated"

    def test_get_tokenization_method_fine_tuned(self):
        """Test getting tokenization method for fine-tuned models."""
        # Fine-tuned OpenAI models should still use tiktoken
        assert (
            get_tokenization_method("gpt-4-turbo-2024-04-09") == "exact-tiktoken"
        )


class TestClearTokenizerCache:
    """Test suite for clear_tokenizer_cache function."""

    def test_clear_tokenizer_cache(self):
        """Test clearing tokenizer cache."""
        # Count tokens to populate cache
        message = {"role": "user", "content": "Test cache clearing."}
        count_tokens(message, model="gpt-4")

        # Clear cache
        clear_tokenizer_cache()

        # Should still work after clearing
        token_count = count_tokens(message, model="gpt-4")
        assert token_count > 0


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_empty_string_content(self):
        """Test with empty string content."""
        message = {"role": "user", "content": ""}
        token_count = count_tokens(message, model="gpt-4")
        assert token_count == 0

    def test_whitespace_only_content(self):
        """Test with whitespace-only content."""
        message = {"role": "user", "content": "   \n\t  "}
        token_count = count_tokens(message, model="gpt-4")
        # Should count whitespace tokens
        assert token_count >= 0

    def test_unicode_content(self):
        """Test with unicode content."""
        message = {"role": "user", "content": "Hello 世界 🌍"}
        token_count = count_tokens(message, model="gpt-4")
        assert token_count > 0

    def test_multimodal_empty_text(self):
        """Test multimodal message with empty text."""
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": ""},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"},
                },
            ],
        }
        token_count = count_tokens(message, model="gpt-4o")
        # Should only count image tokens
        assert token_count > 0

    def test_multimodal_missing_text_field(self):
        """Test multimodal message with missing text field."""
        message = {
            "role": "user",
            "content": [
                {"type": "text"},  # Missing 'text' field
            ],
        }
        token_count = count_tokens(message, model="gpt-4o")
        # Should handle gracefully
        assert token_count == 0

    def test_multimodal_unknown_type(self):
        """Test multimodal message with unknown content type."""
        message = {
            "role": "user",
            "content": [
                {"type": "unknown", "data": "something"},
                {"type": "text", "text": "Hello"},
            ],
        }
        token_count = count_tokens(message, model="gpt-4o")
        # Should only count known types
        assert token_count > 0

    def test_nested_content_structures(self):
        """Test with nested content structures."""
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "First part",
                },
                {
                    "type": "text",
                    "text": "Second part",
                },
            ],
        }
        token_count = count_tokens(message, model="gpt-4")
        # Should count all text parts
        assert token_count > 0

    def test_consistency_across_calls(self):
        """Test that token counting is consistent across calls."""
        message = {"role": "user", "content": "Consistency test message."}

        count1 = count_tokens(message, model="gpt-4")
        count2 = count_tokens(message, model="gpt-4")

        assert count1 == count2


class TestPerformance:
    """Test suite for performance characteristics."""

    def test_tokenizer_caching_works(self):
        """Test that tokenizer caching improves performance."""
        message = {"role": "user", "content": "Test caching performance."}

        # First call loads tokenizer
        count_tokens(message, model="gpt-4")

        # Subsequent calls should use cached tokenizer
        # (We can't easily measure time in unit tests, but we verify it works)
        for _ in range(5):
            token_count = count_tokens(message, model="gpt-4")
            assert token_count > 0

    def test_cache_size_limit(self):
        """Test that cache size is limited (max 10 tokenizers)."""
        # Load more than 10 different models (using estimation for most)
        models = [f"model-{i}" for i in range(15)]

        message = {"role": "user", "content": "Test"}

        for model in models:
            count_tokens(message, model=model, exact=False)

        # Should still work (cache evicts oldest entries)
        token_count = count_tokens(message, model="model-0", exact=False)
        assert token_count > 0
