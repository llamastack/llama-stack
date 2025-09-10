# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from .strong_typing.schema import json_schema_type, register_schema  # noqa: F401


@dataclass
class WebMethod:
    route: str | None = None
    public: bool = False
    request_examples: list[Any] | None = None
    response_examples: list[Any] | None = None
    method: str | None = None
    raw_bytes_request_body: bool | None = False
    # A descriptive name of the corresponding span created by tracing
    descriptive_name: str | None = None
    experimental: bool | None = False
    required_scope: str | None = None


T = TypeVar("T", bound=Callable[..., Any])


def webmethod(
    route: str | None = None,
    method: str | None = None,
    public: bool | None = False,
    request_examples: list[Any] | None = None,
    response_examples: list[Any] | None = None,
    raw_bytes_request_body: bool | None = False,
    descriptive_name: str | None = None,
    experimental: bool | None = False,
    required_scope: str | None = None,
) -> Callable[[T], T]:
    """
    Decorator that supplies additional metadata to an endpoint operation function.

    :param route: The URL path pattern associated with this operation which path parameters are substituted into.
    :param public: True if the operation can be invoked without prior authentication.
    :param request_examples: Sample requests that the operation might take. Pass a list of objects, not JSON.
    :param response_examples: Sample responses that the operation might produce. Pass a list of objects, not JSON.
    :param experimental: True if the operation is experimental and subject to change.
    :param required_scope: Required scope for this endpoint (e.g., 'monitoring.viewer').
    """

    def wrap(func: T) -> T:
        func.__webmethod__ = WebMethod(  # type: ignore
            route=route,
            method=method,
            public=public or False,
            request_examples=request_examples,
            response_examples=response_examples,
            raw_bytes_request_body=raw_bytes_request_body,
            descriptive_name=descriptive_name,
            experimental=experimental,
            required_scope=required_scope,
        )
        return func

    return wrap
