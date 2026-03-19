# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re


def parse_data_url(data_url: str):
    data_url_pattern = re.compile(
        r"^"
        r"data:"
        r"(?P<mimetype>[\w/\-+.]+)"
        r"(?P<charset>;charset=(?P<encoding>[\w-]+))?"
        r"(?P<base64>;base64)?"
        r",(?P<data>.*)"
        r"$",
        re.DOTALL,
    )
    match = data_url_pattern.match(data_url)
    if not match:
        raise ValueError("Invalid Data URL format")

    parts = match.groupdict()
    parts["is_base64"] = bool(parts["base64"])
    return parts
