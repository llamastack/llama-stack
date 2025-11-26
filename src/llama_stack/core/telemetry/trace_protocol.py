# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import inspect
import json
from collections.abc import AsyncGenerator, Callable
from functools import wraps
from typing import Any, cast

from pydantic import BaseModel

from llama_stack.models.llama.datatypes import Primitive

type JSONValue = Primitive | list["JSONValue"] | dict[str, "JSONValue"]


def serialize_value(value: Any) -> str:
    return str(_prepare_for_json(value))


def _prepare_for_json(value: Any) -> JSONValue:
    """Serialize a single value into JSON-compatible format."""
    if value is None:
        return ""
    elif isinstance(value, str | int | float | bool):
        return value
    elif hasattr(value, "_name_"):
        return cast(str, value._name_)
    elif isinstance(value, BaseModel):
        return cast(JSONValue, json.loads(value.model_dump_json()))
    elif isinstance(value, list | tuple | set):
        return [_prepare_for_json(item) for item in value]
    elif isinstance(value, dict):
        return {str(k): _prepare_for_json(v) for k, v in value.items()}
    else:
        try:
            json.dumps(value)
            return cast(JSONValue, value)
        except Exception:
            return str(value)


def trace_protocol[T: type[Any]](cls: T) -> T:
    """
    A class decorator that automatically traces all methods in a protocol/base class
    and its inheriting classes.
    """

    def trace_method(method: Callable[..., Any]) -> Callable[..., Any]:
        is_async = asyncio.iscoroutinefunction(method)
        is_async_gen = inspect.isasyncgenfunction(method)

        def create_span_context(self: Any, *args: Any, **kwargs: Any) -> tuple[str, str, dict[str, Primitive]]:
            class_name = self.__class__.__name__
            method_name = method.__name__
            span_type = "async_generator" if is_async_gen else "async" if is_async else "sync"
            sig = inspect.signature(method)
            param_names = list(sig.parameters.keys())[1:]  # Skip 'self'
            combined_args: dict[str, str] = {}
            for i, arg in enumerate(args):
                param_name = param_names[i] if i < len(param_names) else f"position_{i + 1}"
                combined_args[param_name] = serialize_value(arg)
            for k, v in kwargs.items():
                combined_args[str(k)] = serialize_value(v)

            span_attributes: dict[str, Primitive] = {
                "__autotraced__": True,
                "__class__": class_name,
                "__method__": method_name,
                "__type__": span_type,
                "__args__": json.dumps(combined_args),
            }

            return class_name, method_name, span_attributes

        @wraps(method)
        async def async_gen_wrapper(self: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            from llama_stack.core.telemetry import tracing

            class_name, method_name, span_attributes = create_span_context(self, *args, **kwargs)

            with tracing.span(f"{class_name}.{method_name}", span_attributes) as span:
                count = 0
                try:
                    async for item in method(self, *args, **kwargs):
                        yield item
                        count += 1
                finally:
                    span.set_attribute("chunk_count", count)

        @wraps(method)
        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            from llama_stack.core.telemetry import tracing

            class_name, method_name, span_attributes = create_span_context(self, *args, **kwargs)

            with tracing.span(f"{class_name}.{method_name}", span_attributes) as span:
                try:
                    result = await method(self, *args, **kwargs)
                    span.set_attribute("output", serialize_value(result))
                    return result
                except Exception as e:
                    span.set_attribute("error", str(e))
                    raise

        @wraps(method)
        def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            from llama_stack.core.telemetry import tracing

            class_name, method_name, span_attributes = create_span_context(self, *args, **kwargs)

            with tracing.span(f"{class_name}.{method_name}", span_attributes) as span:
                try:
                    result = method(self, *args, **kwargs)
                    span.set_attribute("output", serialize_value(result))
                    return result
                except Exception as e:
                    span.set_attribute("error", str(e))
                    raise

        if is_async_gen:
            return async_gen_wrapper
        elif is_async:
            return async_wrapper
        else:
            return sync_wrapper

    # Wrap methods on the class itself (for classes applied at runtime)
    # Skip if already wrapped (indicated by __wrapped__ attribute)
    for name, method in vars(cls).items():
        if inspect.isfunction(method) and not name.startswith("_"):
            if not hasattr(method, "__wrapped__"):
                wrapped = trace_method(method)
                setattr(cls, name, wrapped)  # noqa: B010

    # Also set up __init_subclass__ for future subclasses
    original_init_subclass = cast(Callable[..., Any] | None, getattr(cls, "__init_subclass__", None))

    def __init_subclass__(cls_child: type[Any], **kwargs: Any) -> None:  # noqa: N807
        if original_init_subclass:
            cast(Callable[..., None], original_init_subclass)(**kwargs)

        for name, method in vars(cls_child).items():
            if inspect.isfunction(method) and not name.startswith("_"):
                setattr(cls_child, name, trace_method(method))  # noqa: B010

    cls_any = cast(Any, cls)
    cls_any.__init_subclass__ = classmethod(__init_subclass__)

    return cls
