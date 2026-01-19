# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Validators for API request parameters.

This module contains validation functions used by providers to validate
request parameters that cannot be easily validated using Pydantic alone.
"""


def validate_embedding_input_is_text(
    input: str | list[str] | list[int] | list[list[int]],
    provider_name: str,
) -> None:
    """
    Validate that embedding input contains only text strings, not token arrays.

    Token arrays (list[int] and list[list[int]]) are a newer OpenAI feature
    that is not universally supported across all embedding providers. This
    validator should be called by providers that only support text input.

    :param input: The embedding input to validate
    :param provider_name: Name of the provider for error message context
    :raises ValueError: If input contains token arrays

    Example usage in provider:
        validate_embedding_input_is_text(params.input, "sentence-transformers")
    """
    # Valid: string input
    if isinstance(input, str):
        return

    # Valid: list of strings
    if isinstance(input, list) and len(input) > 0 and isinstance(input[0], str):
        return

    # If we get here, input is a token array (list[int] or list[list[int]])
    raise ValueError(
        f"Provider '{provider_name}' does not support token arrays. "
        f"Token arrays are currently only supported by OpenAI and Fireworks providers. "
        f"Please provide text input as a string or list of strings instead."
    )


__all__ = [
    "validate_embedding_input_is_text",
]
