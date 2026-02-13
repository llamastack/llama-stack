# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import httpx
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from llama_stack.core.exceptions.mapping import translate_exception_to_http
from llama_stack_api.common.errors import LlamaStackError


def translate_exception(exc: Exception) -> HTTPException:
    """Translate an exception to an HTTPException."""
    if isinstance(exc, ValidationError):
        exc = RequestValidationError(exc.errors())

    if isinstance(exc, RequestValidationError):
        return HTTPException(
            status_code=httpx.codes.BAD_REQUEST,
            detail={
                "errors": [
                    {
                        "loc": list(error.get("loc", [])),
                        "msg": error.get("msg", "Validation error"),
                        "type": error.get("type", "unknown"),
                    }
                    for error in exc.errors()
                ]
            },
        )

    if isinstance(exc, LlamaStackError):
        return HTTPException(status_code=exc.status_code, detail=str(exc))

    # Translate generic exceptions to HTTPException
    http_exc = translate_exception_to_http(exc)
    if http_exc:
        return http_exc

    if hasattr(exc, "status_code") and isinstance(getattr(exc, "status_code", None), int):
        # Handle provider SDK exceptions (e.g., OpenAI's APIStatusError and subclasses)
        # These include AuthenticationError (401), PermissionDeniedError (403), etc.
        # This preserves the actual HTTP status code from the provider
        status_code = getattr(exc, "status_code", httpx.codes.INTERNAL_SERVER_ERROR)
        detail = str(exc)
        return HTTPException(status_code=status_code, detail=detail)
    else:
        return HTTPException(
            status_code=httpx.codes.INTERNAL_SERVER_ERROR,
            detail="Internal server error: An unexpected error occurred.",
        )
