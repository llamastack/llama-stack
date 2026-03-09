# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared header-filtering utilities for provider data forwarding.

The safety passthrough provider (remote/safety/passthrough/config.py) maintains
its own _BLOCKED_HEADERS set for config-time validation of forward_headers.
A future follow-up could migrate it to use this shared module.
"""

# Hop-by-hop, framing, and security-sensitive headers that must never be
# forwarded to upstream providers via extra_headers.
#
# This is a superset of the safety passthrough's blocked list — it adds
# auth and proxy headers because extra_headers is caller-controlled
# (per-request), unlike the deployer-controlled forward_headers config.
BLOCKED_HEADERS: frozenset[str] = frozenset(
    {
        # Hop-by-hop / framing (also in safety passthrough's list)
        "host",
        "content-type",
        "content-length",
        "transfer-encoding",
        "connection",
        "upgrade",
        "te",
        "trailer",
        "cookie",
        "set-cookie",
        # Auth — prevent callers from overriding the provider's Authorization
        "authorization",
        # Proxy / origin — prevent spoofing
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
        "x-forwarded-prefix",
        "x-real-ip",
        "cf-connecting-ip",
        "true-client-ip",
    }
)


def filter_extra_headers(headers: dict[str, str] | None) -> dict[str, str] | None:
    """Filter out blocked headers from a caller-supplied extra_headers dict.

    Returns the filtered dict, or ``None`` if *headers* is ``None``, not a
    dict, or no headers remain after filtering.
    """
    if not isinstance(headers, dict):
        return None
    filtered = {k: v for k, v in headers.items() if k.lower() not in BLOCKED_HEADERS}
    return filtered if filtered else None
