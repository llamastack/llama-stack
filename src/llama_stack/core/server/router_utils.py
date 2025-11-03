# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Utilities for creating FastAPI routers with standard error responses."""

from llama_stack.apis.datatypes import Error

standard_responses = {
    400: {
        "model": Error,
        "description": "The request was invalid or malformed.",
    },
    429: {
        "model": Error,
        "description": "The client has sent too many requests in a given amount of time.",
    },
    500: {
        "model": Error,
        "description": "The server encountered an unexpected error.",
    },
    "default": {
        "model": Error,
        "description": "An unexpected error occurred.",
    },
}
