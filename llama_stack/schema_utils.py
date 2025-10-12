# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from .strong_typing.schema import json_schema_type, register_schema  # noqa: F401


class ExtraBodyField[T]:
    """
    Marker annotation for parameters that arrive via extra_body in the client SDK.

    These parameters:
    - Will NOT appear in the generated client SDK method signature
    - WILL be documented in OpenAPI spec under x-llama-stack-extra-body-params
    - MUST be passed via the extra_body parameter in client SDK calls
    - WILL be available in server-side method signature with proper typing

    Example:
        ```python
        async def create_openai_response(
            self,
            input: str,
            model: str,
            shields: Annotated[
                list[str] | None, ExtraBodyField("List of shields to apply")
            ] = None,
        ) -> ResponseObject:
            # shields is available here with proper typing
            if shields:
                print(f"Using shields: {shields}")
        ```

        Client usage:
        ```python
        client.responses.create(
            input="hello", model="llama-3", extra_body={"shields": ["shield-1"]}
        )
        ```
    """

    def __init__(self, description: str | None = None):
        self.description = description


@dataclass
class WebMethod:
    level: str | None = None
    route: str | None = None
    public: bool = False
    request_examples: list[Any] | None = None
    response_examples: list[Any] | None = None
    method: str | None = None
    raw_bytes_request_body: bool | None = False
    # A descriptive name of the corresponding span created by tracing
    descriptive_name: str | None = None
    required_scope: str | None = None
    deprecated: bool | None = False
    require_authentication: bool | None = True


CallableT = TypeVar("CallableT", bound=Callable[..., Any])


def webmethod(
    route: str | None = None,
    method: str | None = None,
    level: str | None = None,
    public: bool | None = False,
    request_examples: list[Any] | None = None,
    response_examples: list[Any] | None = None,
    raw_bytes_request_body: bool | None = False,
    descriptive_name: str | None = None,
    required_scope: str | None = None,
    deprecated: bool | None = False,
    require_authentication: bool | None = True,
) -> Callable[[CallableT], CallableT]:
    """
    Decorator that supplies additional metadata to an endpoint operation function.

    :param route: The URL path pattern associated with this operation which path parameters are substituted into.
    :param public: True if the operation can be invoked without prior authentication.
    :param request_examples: Sample requests that the operation might take. Pass a list of objects, not JSON.
    :param response_examples: Sample responses that the operation might produce. Pass a list of objects, not JSON.
    :param required_scope: Required scope for this endpoint (e.g., 'monitoring.viewer').
    :param require_authentication: Whether this endpoint requires authentication (default True).
    """

    def wrap(func: CallableT) -> CallableT:
        webmethod_obj = WebMethod(
            route=route,
            method=method,
            level=level,
            public=public or False,
            request_examples=request_examples,
            response_examples=response_examples,
            raw_bytes_request_body=raw_bytes_request_body,
            descriptive_name=descriptive_name,
            required_scope=required_scope,
            deprecated=deprecated,
            require_authentication=require_authentication if require_authentication is not None else True,
        )

        # Store all webmethods in a list to support multiple decorators
        if not hasattr(func, "__webmethods__"):
            func.__webmethods__ = []  # type: ignore
        func.__webmethods__.append(webmethod_obj)  # type: ignore

        # Keep the last one as __webmethod__ for backwards compatibility
        func.__webmethod__ = webmethod_obj  # type: ignore
        return func

    return wrap
