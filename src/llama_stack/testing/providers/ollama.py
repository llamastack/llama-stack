# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Ollama provider exception handling for test recording/replay.

Handles native Ollama errors that don't go through the OpenAI SDK.
Most Ollama inference errors go through OpenAIMixin and are already
handled as OpenAI errors.
"""

from ollama import ResponseError

# Provider configuration
NAME = "ollama"
MODULE_PREFIX = "ollama"


def create_error(status_code: int, body: dict | None, message: str) -> ResponseError:
    """Reconstruct an Ollama ResponseError from recorded data."""
    # Ollama's ResponseError only takes error message and status_code
    # It doesn't have a body attribute like OpenAI errors
    return ResponseError(error=message, status_code=status_code)
