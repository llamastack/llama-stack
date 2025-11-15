# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Token counting utilities for prompt caching.

This module provides token counting functionality for various model families,
supporting exact tokenization for OpenAI and Llama models, with fallback
estimation for unknown models.
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from llama_stack.log import get_logger

logger = get_logger(__name__)


# Model family patterns for exact tokenization
OPENAI_MODELS = {
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-3.5-turbo",
    "o1-preview",
    "o1-mini",
}

LLAMA_MODEL_PREFIXES = [
    "meta-llama/Llama-3",
    "meta-llama/Llama-4",
    "meta-llama/Meta-Llama-3",
]

# Default estimation parameters
DEFAULT_CHARS_PER_TOKEN = 4  # Conservative estimate for unknown models
DEFAULT_IMAGE_TOKENS_LOW_RES = 85  # GPT-4V low-res image token estimate
DEFAULT_IMAGE_TOKENS_HIGH_RES = 170  # GPT-4V high-res image token estimate


class TokenizationError(Exception):
    """Exception raised for tokenization errors."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        """Initialize tokenization error.

        Args:
            message: Error description (should start with "Failed to ...")
            cause: Optional underlying exception that caused this error
        """
        super().__init__(message)
        self.cause = cause


@lru_cache(maxsize=10)
def _get_tiktoken_encoding(model: str):
    """Get tiktoken encoding for OpenAI models.

    Args:
        model: OpenAI model name

    Returns:
        Tiktoken encoding instance

    Raises:
        TokenizationError: If encoding cannot be loaded
    """
    try:
        import tiktoken

        # Try to get encoding for specific model
        try:
            encoding = tiktoken.encoding_for_model(model)
            logger.debug(f"Loaded tiktoken encoding for model: {model}")
            return encoding
        except KeyError:
            # Fall back to cl100k_base for GPT-4 and newer models
            logger.debug(f"No specific encoding for {model}, using cl100k_base")
            return tiktoken.get_encoding("cl100k_base")

    except ImportError as e:
        raise TokenizationError(
            f"Failed to import tiktoken for model {model}. "
            "Install with: pip install tiktoken",
            cause=e,
        ) from e
    except Exception as e:
        raise TokenizationError(
            f"Failed to load tiktoken encoding for model {model}",
            cause=e,
        ) from e


@lru_cache(maxsize=10)
def _get_transformers_tokenizer(model: str):
    """Get HuggingFace transformers tokenizer for Llama models.

    Args:
        model: Llama model name or path

    Returns:
        Transformers tokenizer instance

    Raises:
        TokenizationError: If tokenizer cannot be loaded
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model)
        logger.debug(f"Loaded transformers tokenizer for model: {model}")
        return tokenizer

    except ImportError as e:
        raise TokenizationError(
            f"Failed to import transformers for model {model}. "
            "Install with: pip install transformers",
            cause=e,
        ) from e
    except Exception as e:
        raise TokenizationError(
            f"Failed to load transformers tokenizer for model {model}",
            cause=e,
        ) from e


def _is_openai_model(model: str) -> bool:
    """Check if model is an OpenAI model.

    Args:
        model: Model name

    Returns:
        True if OpenAI model, False otherwise
    """
    # Check exact matches
    if model in OPENAI_MODELS:
        return True

    # Check prefixes (for fine-tuned models like gpt-4-turbo-2024-04-09)
    for base_model in OPENAI_MODELS:
        if model.startswith(base_model):
            return True

    return False


def _is_llama_model(model: str) -> bool:
    """Check if model is a Llama model.

    Args:
        model: Model name

    Returns:
        True if Llama model, False otherwise
    """
    for prefix in LLAMA_MODEL_PREFIXES:
        if model.startswith(prefix):
            return True
    return False


def _count_tokens_openai(text: str, model: str) -> int:
    """Count tokens using tiktoken for OpenAI models.

    Args:
        text: Text to count tokens for
        model: OpenAI model name

    Returns:
        Number of tokens

    Raises:
        TokenizationError: If tokenization fails
    """
    try:
        encoding = _get_tiktoken_encoding(model)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        if isinstance(e, TokenizationError):
            raise
        raise TokenizationError(
            f"Failed to count tokens for OpenAI model {model}",
            cause=e,
        ) from e


def _count_tokens_llama(text: str, model: str) -> int:
    """Count tokens using transformers for Llama models.

    Args:
        text: Text to count tokens for
        model: Llama model name

    Returns:
        Number of tokens

    Raises:
        TokenizationError: If tokenization fails
    """
    try:
        tokenizer = _get_transformers_tokenizer(model)
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    except Exception as e:
        if isinstance(e, TokenizationError):
            raise
        raise TokenizationError(
            f"Failed to count tokens for Llama model {model}",
            cause=e,
        ) from e


def _estimate_tokens_from_chars(text: str) -> int:
    """Estimate token count from character count.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated number of tokens
    """
    return max(1, len(text) // DEFAULT_CHARS_PER_TOKEN)


def _count_tokens_for_text(text: str, model: str, exact: bool = True) -> int:
    """Count tokens for text content.

    Args:
        text: Text to count tokens for
        model: Model name
        exact: If True, use exact tokenization; if False, estimate

    Returns:
        Number of tokens
    """
    if not text:
        return 0

    # Use exact tokenization if requested
    if exact:
        try:
            if _is_openai_model(model):
                return _count_tokens_openai(text, model)
            elif _is_llama_model(model):
                return _count_tokens_llama(text, model)
        except TokenizationError as e:
            logger.warning(
                f"Failed to get exact token count for model {model}, "
                f"falling back to estimation: {e}"
            )

    # Fall back to estimation
    return _estimate_tokens_from_chars(text)


def _count_tokens_for_image(
    image_content: Dict[str, Any],
    model: str,
) -> int:
    """Estimate token count for image content.

    Args:
        image_content: Image content dictionary with 'image_url' or 'detail'
        model: Model name

    Returns:
        Estimated number of tokens for the image
    """
    # For now, use GPT-4V estimates as baseline
    # Future: could add model-specific image token calculations

    detail = "auto"
    if isinstance(image_content, dict):
        # Check for detail in image_url
        image_url = image_content.get("image_url", {})
        if isinstance(image_url, dict):
            detail = image_url.get("detail", "auto")

    # Estimate based on detail level
    if detail == "low":
        return DEFAULT_IMAGE_TOKENS_LOW_RES
    elif detail == "high":
        return DEFAULT_IMAGE_TOKENS_HIGH_RES
    else:  # "auto" or unknown
        # Use average of low and high
        return (DEFAULT_IMAGE_TOKENS_LOW_RES + DEFAULT_IMAGE_TOKENS_HIGH_RES) // 2


def _count_tokens_for_message(
    message: Dict[str, Any],
    model: str,
    exact: bool = True,
) -> int:
    """Count tokens for a single message.

    Args:
        message: Message dictionary with 'role' and 'content'
        model: Model name
        exact: If True, use exact tokenization for text

    Returns:
        Total number of tokens in the message
    """
    total_tokens = 0

    # Handle None or malformed messages
    if not message or not isinstance(message, dict):
        return 0

    content = message.get("content")

    # Handle empty content
    if content is None:
        return 0

    # Handle string content (simple text message)
    if isinstance(content, str):
        return _count_tokens_for_text(content, model, exact=exact)

    # Handle list content (multimodal message)
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")

            if item_type == "text":
                text = item.get("text", "")
                total_tokens += _count_tokens_for_text(text, model, exact=exact)

            elif item_type == "image_url":
                total_tokens += _count_tokens_for_image(item, model)

    return total_tokens


def count_tokens(
    messages: Union[List[Dict[str, Any]], Dict[str, Any]],
    model: str,
    exact: bool = True,
) -> int:
    """Count total tokens in messages for a given model.

    This function supports:
    - Exact tokenization for OpenAI models (using tiktoken)
    - Exact tokenization for Llama models (using transformers)
    - Character-based estimation for unknown models
    - Multimodal content (text + images)

    Args:
        messages: Single message or list of messages to count tokens for.
                 Each message should have 'role' and 'content' fields.
        model: Model name (e.g., "gpt-4", "meta-llama/Llama-3.1-8B-Instruct")
        exact: If True, use exact tokenization where available.
               If False or if exact tokenization fails, use estimation.

    Returns:
        Total number of tokens across all messages

    Raises:
        TokenizationError: If tokenization fails and fallback also fails

    Examples:
        >>> # Single text message
        >>> count_tokens(
        ...     {"role": "user", "content": "Hello, world!"},
        ...     model="gpt-4"
        ... )
        4

        >>> # Multiple messages
        >>> count_tokens(
        ...     [
        ...         {"role": "system", "content": "You are a helpful assistant."},
        ...         {"role": "user", "content": "What is the weather?"}
        ...     ],
        ...     model="gpt-4"
        ... )
        15

        >>> # Multimodal message with image
        >>> count_tokens(
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "text", "text": "What's in this image?"},
        ...             {"type": "image_url", "image_url": {"url": "...", "detail": "low"}}
        ...         ]
        ...     },
        ...     model="gpt-4o"
        ... )
        90
    """
    # Handle single message
    if isinstance(messages, dict):
        return _count_tokens_for_message(messages, model, exact=exact)

    # Handle list of messages
    if not isinstance(messages, list):
        logger.warning(f"Invalid messages type: {type(messages)}, returning 0")
        return 0

    total_tokens = 0
    for message in messages:
        total_tokens += _count_tokens_for_message(message, model, exact=exact)

    return total_tokens


def get_tokenization_method(model: str) -> str:
    """Get the tokenization method used for a model.

    Args:
        model: Model name

    Returns:
        Tokenization method: "exact-tiktoken", "exact-transformers", or "estimated"

    Examples:
        >>> get_tokenization_method("gpt-4")
        'exact-tiktoken'
        >>> get_tokenization_method("meta-llama/Llama-3.1-8B-Instruct")
        'exact-transformers'
        >>> get_tokenization_method("unknown-model")
        'estimated'
    """
    if _is_openai_model(model):
        return "exact-tiktoken"
    elif _is_llama_model(model):
        return "exact-transformers"
    else:
        return "estimated"


def clear_tokenizer_cache() -> None:
    """Clear the tokenizer cache.

    This is useful for testing or when you want to free up memory.
    """
    _get_tiktoken_encoding.cache_clear()
    _get_transformers_tokenizer.cache_clear()
    logger.info("Tokenizer cache cleared")
