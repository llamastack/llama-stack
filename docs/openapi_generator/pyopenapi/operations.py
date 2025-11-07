# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import collections.abc
import enum
import inspect
import typing
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from llama_stack.apis.version import LLAMA_STACK_API_V1, LLAMA_STACK_API_V1BETA, LLAMA_STACK_API_V1ALPHA

from termcolor import colored

from llama_stack.strong_typing.inspection import get_signature

from typing import get_origin, get_args

from fastapi import UploadFile
from fastapi.params import File, Form
from typing import Annotated

from llama_stack.schema_utils import ExtraBodyField


def split_prefix(
    s: str, sep: str, prefix: Union[str, Iterable[str]]
) -> Tuple[Optional[str], str]:
    """
    Recognizes a prefix at the beginning of a string.

    :param s: The string to check.
    :param sep: A separator between (one of) the prefix(es) and the rest of the string.
    :param prefix: A string or a set of strings to identify as a prefix.
    :return: A tuple of the recognized prefix (if any) and the rest of the string excluding the separator (or the entire string).
    """

    if isinstance(prefix, str):
        if s.startswith(prefix + sep):
            return prefix, s[len(prefix) + len(sep) :]
        else:
            return None, s

    for p in prefix:
        if s.startswith(p + sep):
            return p, s[len(p) + len(sep) :]

    return None, s


def _get_annotation_type(annotation: Union[type, str], callable: Callable) -> type:
    "Maps a stringized reference to a type, as if using `from __future__ import annotations`."

    if isinstance(annotation, str):
        return eval(annotation, callable.__globals__)
    else:
        return annotation


class HTTPMethod(enum.Enum):
    "HTTP method used to invoke an endpoint operation."

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


OperationParameter = Tuple[str, type]


class ValidationError(TypeError):
    pass


@dataclass
class EndpointOperation:
    """
    Type information and metadata associated with an endpoint operation.

    "param defining_class: The most specific class that defines the endpoint operation.
    :param name: The short name of the endpoint operation.
    :param func_name: The name of the function to invoke when the operation is triggered.
    :param func_ref: The callable to invoke when the operation is triggered.
    :param route: A custom route string assigned to the operation.
    :param path_params: Parameters of the operation signature that are passed in the path component of the URL string.
    :param query_params: Parameters of the operation signature that are passed in the query string as `key=value` pairs.
    :param request_params: The parameter that corresponds to the data transmitted in the request body.
    :param multipart_params: Parameters that indicate multipart/form-data request body.
    :param extra_body_params: Parameters that arrive via extra_body and are documented but not in SDK.
    :param event_type: The Python type of the data that is transmitted out-of-band (e.g. via websockets) while the operation is in progress.
    :param response_type: The Python type of the data that is transmitted in the response body.
    :param http_method: The HTTP method used to invoke the endpoint such as POST, GET or PUT.
    :param public: True if the operation can be invoked without prior authentication.
    :param request_examples: Sample requests that the operation might take.
    :param response_examples: Sample responses that the operation might produce.
    """

    defining_class: type
    name: str
    func_name: str
    func_ref: Callable[..., Any]
    route: Optional[str]
    path_params: List[OperationParameter]
    query_params: List[OperationParameter]
    request_params: Optional[OperationParameter]
    multipart_params: List[OperationParameter]
    extra_body_params: List[tuple[str, type, str | None]]
    event_type: Optional[type]
    response_type: type
    http_method: HTTPMethod
    public: bool
    request_examples: Optional[List[Any]] = None
    response_examples: Optional[List[Any]] = None

    def get_route(self, webmethod) -> str:
        api_level = webmethod.level

        if self.route is not None:
            return "/".join(["", api_level, self.route.lstrip("/")])

        route_parts = ["", api_level, self.name]
        for param_name, _ in self.path_params:
            route_parts.append("{" + param_name + "}")
        return "/".join(route_parts)


class _FormatParameterExtractor:
    "A visitor to exract parameters in a format string."

    keys: List[str]

    def __init__(self) -> None:
        self.keys = []

    def __getitem__(self, key: str) -> None:
        self.keys.append(key)
        return None


def _get_route_parameters(route: str) -> List[str]:
    extractor = _FormatParameterExtractor()
    # Replace all occurrences of ":path" with empty string
    route = route.replace(":path", "")
    route.format_map(extractor)
    return extractor.keys


def _get_endpoint_functions(
    endpoint: type, prefixes: List[str]
) -> Iterator[Tuple[str, str, str, Callable]]:
    if not inspect.isclass(endpoint):
        raise ValueError(f"object is not a class type: {endpoint}")

    functions = inspect.getmembers(endpoint, inspect.isfunction)
    for func_name, func_ref in functions:
        webmethods = []

        # Check for multiple webmethods (stacked decorators)
        if hasattr(func_ref, "__webmethods__"):
            webmethods = func_ref.__webmethods__

        if not webmethods:
            continue

        for webmethod in webmethods:
            print(f"Processing {colored(func_name, 'white')}...")
            operation_name = func_name

            if webmethod.method == "GET":
                prefix = "get"
            elif webmethod.method == "DELETE":
                prefix = "delete"
            elif webmethod.method == "POST":
                prefix = "post"
            elif operation_name.startswith("get_") or operation_name.endswith("/get"):
                prefix = "get"
            elif (
                operation_name.startswith("delete_")
                or operation_name.startswith("remove_")
                or operation_name.endswith("/delete")
                or operation_name.endswith("/remove")
            ):
                prefix = "delete"
            else:
                # by default everything else is a POST
                prefix = "post"

            yield prefix, operation_name, func_name, func_ref


def _get_defining_class(member_fn: str, derived_cls: type) -> type:
    "Find the class in which a member function is first defined in a class inheritance hierarchy."

    # iterate in reverse member resolution order to find most specific class first
    for cls in reversed(inspect.getmro(derived_cls)):
        for name, _ in inspect.getmembers(cls, inspect.isfunction):
            if name == member_fn:
                return cls

    raise ValidationError(
        f"cannot find defining class for {member_fn} in {derived_cls}"
    )


def get_endpoint_operations(
    endpoint: type, use_examples: bool = True
) -> List[EndpointOperation]:
    """
    Extracts a list of member functions in a class eligible for HTTP interface binding.

    These member functions are expected to have a signature like
    ```
    async def get_object(self, uuid: str, version: int) -> Object:
        ...
    ```
    where the prefix `get_` translates to an HTTP GET, `object` corresponds to the name of the endpoint operation,
    `uuid` and `version` are mapped to route path elements in "/object/{uuid}/{version}", and `Object` becomes
    the response payload type, transmitted as an object serialized to JSON.

    If the member function has a composite class type in the argument list, it becomes the request payload type,
    and the caller is expected to provide the data as serialized JSON in an HTTP POST request.

    :param endpoint: A class with member functions that can be mapped to an HTTP endpoint.
    :param use_examples: Whether to return examples associated with member functions.
    """

    result = []

    for prefix, operation_name, func_name, func_ref in _get_endpoint_functions(
        endpoint,
        [
            "create",
            "delete",
            "do",
            "get",
            "post",
            "put",
            "remove",
            "set",
            "update",
        ],
    ):
        # Get all webmethods for this function
        webmethods = getattr(func_ref, "__webmethods__", [])

        # Create one EndpointOperation for each webmethod
        for webmethod in webmethods:
            route = webmethod.route
            route_params = _get_route_parameters(route) if route is not None else None
            public = webmethod.public
            request_examples = webmethod.request_examples
            response_examples = webmethod.response_examples

            # inspect function signature for path and query parameters, and request/response payload type
            signature = get_signature(func_ref)

            path_params = []
            query_params = []
            request_params = []
            multipart_params = []
            extra_body_params = []

            for param_name, parameter in signature.parameters.items():
                param_type = _get_annotation_type(parameter.annotation, func_ref)

                # omit "self" for instance methods
                if param_name == "self" and param_type is inspect.Parameter.empty:
                    continue

                # check if all parameters have explicit type
                if parameter.annotation is inspect.Parameter.empty:
                    raise ValidationError(
                        f"parameter '{param_name}' in function '{func_name}' has no type annotation"
                    )

                # Check if this is an extra_body parameter
                is_extra_body, extra_body_desc = _is_extra_body_param(param_type)
                if is_extra_body:
                    # Store in a separate list for documentation
                    extra_body_params.append((param_name, param_type, extra_body_desc))
                    continue  # Skip adding to request_params

                is_multipart = _is_multipart_param(param_type)

                if prefix in ["get", "delete"]:
                    if route_params is not None and param_name in route_params:
                        path_params.append((param_name, param_type))
                    else:
                        query_params.append((param_name, param_type))
                else:
                    if route_params is not None and param_name in route_params:
                        path_params.append((param_name, param_type))
                    elif is_multipart:
                        multipart_params.append((param_name, param_type))
                    else:
                        request_params.append((param_name, param_type))

            # check if function has explicit return type
            if signature.return_annotation is inspect.Signature.empty:
                raise ValidationError(
                    f"function '{func_name}' has no return type annotation"
                )

            return_type = _get_annotation_type(signature.return_annotation, func_ref)

            # operations that produce events are labeled as Generator[YieldType, SendType, ReturnType]
            # where YieldType is the event type, SendType is None, and ReturnType is the immediate response type to the request
            if typing.get_origin(return_type) is collections.abc.Generator:
                event_type, send_type, response_type = typing.get_args(return_type)
                if send_type is not type(None):
                    raise ValidationError(
                        f"function '{func_name}' has a return type Generator[Y,S,R] and therefore looks like an event but has an explicit send type"
                    )
            else:
                event_type = None

                def process_type(t):
                    if typing.get_origin(t) is collections.abc.AsyncIterator:
                        # NOTE(ashwin): this is SSE and there is no way to represent it. either we make it a List
                        # or the item type. I am choosing it to be the latter
                        args = typing.get_args(t)
                        return args[0]
                    elif typing.get_origin(t) is typing.Union:
                        types = [process_type(a) for a in typing.get_args(t)]
                        return typing._UnionGenericAlias(typing.Union, tuple(types))
                    else:
                        return t

                response_type = process_type(return_type)

                if prefix in ["delete", "remove"]:
                    http_method = HTTPMethod.DELETE
                elif prefix == "post":
                    http_method = HTTPMethod.POST
                elif prefix == "get":
                    http_method = HTTPMethod.GET
                elif prefix == "set":
                    http_method = HTTPMethod.PUT
                elif prefix == "update":
                    http_method = HTTPMethod.PATCH
                else:
                    raise ValidationError(f"unknown prefix {prefix}")

            # Create an EndpointOperation for this specific webmethod
            operation = EndpointOperation(
                defining_class=_get_defining_class(func_name, endpoint),
                name=operation_name,
                func_name=func_name,
                func_ref=func_ref,
                route=route,
                path_params=path_params,
                query_params=query_params,
                request_params=request_params,
                multipart_params=multipart_params,
                extra_body_params=extra_body_params,
                event_type=event_type,
                response_type=response_type,
                http_method=http_method,
                public=public,
                request_examples=request_examples if use_examples else None,
                response_examples=response_examples if use_examples else None,
            )

            # Store the specific webmethod with this operation
            operation.webmethod = webmethod
            result.append(operation)

    if not result:
        raise ValidationError(f"no eligible endpoint operations in type {endpoint}")

    return result


def get_endpoint_events(endpoint: type) -> Dict[str, type]:
    results = {}

    for decl in typing.get_type_hints(endpoint).values():
        # check if signature is Callable[...]
        origin = typing.get_origin(decl)
        if origin is None or not issubclass(origin, Callable):  # type: ignore
            continue

        # check if signature is Callable[[...], Any]
        args = typing.get_args(decl)
        if len(args) != 2:
            continue
        params_type, return_type = args
        if not isinstance(params_type, list):
            continue

        # check if signature is Callable[[...], None]
        if not issubclass(return_type, type(None)):
            continue

        # check if signature is Callable[[EventType], None]
        if len(params_type) != 1:
            continue

        param_type = params_type[0]
        results[param_type.__name__] = param_type

    return results


def _is_multipart_param(param_type: type) -> bool:
    """
    Check if a parameter type indicates multipart form data.

    Returns True if the type is:
    - UploadFile
    - Annotated[UploadFile, File()]
    - Annotated[str, Form()]
    - Annotated[Any, File()]
    - Annotated[Any, Form()]
    """
    if param_type is UploadFile:
        return True

    # Check for Annotated types
    origin = get_origin(param_type)
    if origin is None:
        return False

    if origin is Annotated:
        args = get_args(param_type)
        if len(args) < 2:
            return False

        # Check the annotations for File() or Form()
        for annotation in args[1:]:
            if isinstance(annotation, (File, Form)):
                return True
    return False


def _is_extra_body_param(param_type: type) -> tuple[bool, str | None]:
    """
    Check if parameter is marked as coming from extra_body.

    Returns:
        (is_extra_body, description): Tuple of boolean and optional description
    """
    origin = get_origin(param_type)
    if origin is Annotated:
        args = get_args(param_type)
        for annotation in args[1:]:
            if isinstance(annotation, ExtraBodyField):
                return True, annotation.description
            # Also check by type name for cases where import matters
            if type(annotation).__name__ == 'ExtraBodyField':
                return True, getattr(annotation, 'description', None)
    return False, None
