# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import base64
import io
from urllib.parse import unquote

from llama_stack.providers.utils.memory.vector_store import parse_data_url


async def get_dataframe_from_uri(uri: str):
    import pandas

    df = None
    if uri.endswith(".csv"):
        # Moving to its own thread to avoid io from blocking the eventloop
        # This isn't ideal as it moves more then just the IO to a new thread
        # but it is as close as we can easly get
        df = await asyncio.to_thread(pandas.read_csv, uri)
    elif uri.endswith(".xlsx"):
        df = await asyncio.to_thread(pandas.read_excel, uri)
    elif uri.startswith("data:"):
        parts = parse_data_url(uri)
        data = parts["data"]
        if parts["is_base64"]:
            data = base64.b64decode(data)
        else:
            data = unquote(data)
            encoding = parts["encoding"] or "utf-8"
            data = data.encode(encoding)

        mime_type = parts["mimetype"]
        mime_category = mime_type.split("/")[0]
        data_bytes = io.BytesIO(data)

        if mime_category == "text":
            df = pandas.read_csv(data_bytes)
        else:
            df = pandas.read_excel(data_bytes)
    else:
        raise ValueError(f"Unsupported file type: {uri}")

    return df
