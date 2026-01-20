# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared exception mappings for HTTP status code translation.

This module provides a single source of truth for mapping exception types
to HTTP status codes. It is used by:
- server.py: to translate exceptions to HTTPException responses
- testing/exception_utils.py: to reconstruct exceptions during test replay
"""

import asyncio

import httpx
from fastapi import HTTPException
from openai import BadRequestError

from llama_stack.core.access_control.access_control import AccessDeniedError
from llama_stack.core.datatypes import AuthenticationRequiredError

# Maps exception type -> (status_code, detail_template)
# Use {e} in template to insert the exception message
EXCEPTION_MAP: dict[type, tuple[int, str]] = {
    ValueError: (httpx.codes.BAD_REQUEST, "Invalid value: {e}"),
    BadRequestError: (httpx.codes.BAD_REQUEST, "{e}"),
    PermissionError: (httpx.codes.FORBIDDEN, "Permission denied: {e}"),
    AccessDeniedError: (httpx.codes.FORBIDDEN, "Permission denied: {e}"),
    ConnectionError: (httpx.codes.BAD_GATEWAY, "{e}"),
    httpx.ConnectError: (httpx.codes.BAD_GATEWAY, "{e}"),
    TimeoutError: (httpx.codes.GATEWAY_TIMEOUT, "Operation timed out: {e}"),
    asyncio.TimeoutError: (httpx.codes.GATEWAY_TIMEOUT, "Operation timed out: {e}"),
    NotImplementedError: (httpx.codes.NOT_IMPLEMENTED, "Not implemented: {e}"),
    AuthenticationRequiredError: (httpx.codes.UNAUTHORIZED, "Authentication required: {e}"),
}

# For deserialization by class name (used by testing/exception_utils.py)
EXCEPTION_TYPES_BY_NAME: dict[str, type[Exception]] = {cls.__name__: cls for cls in EXCEPTION_MAP}


def translate_exception_to_http(exc: Exception) -> HTTPException | None:
    """Translate an exception to an HTTPException using the mapping.

    Walks up the exception's inheritance chain (MRO) and checks for a match
    in the mapping. This is O(k) where k is the inheritance depth, with O(1)
    dict lookup at each level.

    Returns None if the exception type is not in the mapping.
    """
    for cls in type(exc).__mro__:
        if cls in EXCEPTION_MAP:
            status_code, template = EXCEPTION_MAP[cls]
            return HTTPException(status_code=status_code, detail=template.format(e=exc))
    return None
