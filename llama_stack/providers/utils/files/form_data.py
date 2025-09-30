# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from fastapi import Request

from llama_stack.apis.files import ExpiresAfter


async def parse_expires_after(request: Request) -> ExpiresAfter | None:
    """
    Dependency to parse expires_after from multipart form data.
    Handles both bracket notation (expires_after[anchor], expires_after[seconds])
    and JSON string format.
    """
    form = await request.form()

    # Check for bracket notation first
    anchor_key = "expires_after[anchor]"
    seconds_key = "expires_after[seconds]"

    if anchor_key in form and seconds_key in form:
        anchor = form[anchor_key]
        seconds = form[seconds_key]
        return ExpiresAfter(anchor=anchor, seconds=int(seconds))

    # Check for JSON string format
    if "expires_after" in form:
        value = form["expires_after"]
        if isinstance(value, str):
            import json

            try:
                data = json.loads(value)
                return ExpiresAfter(**data)
            except (json.JSONDecodeError, TypeError):
                pass

    return None
