# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Type-safe data interchange for Python data classes.

:see: https://github.com/hunyadi/strong_typing
"""

import inspect
import json
import sys
from types import ModuleType
from typing import Any, TextIO, TypeVar

from .core import JsonType
from .deserializer import create_deserializer
from .inspection import TypeLike
from .serializer import create_serializer

T = TypeVar("T")


def object_to_json(obj: Any) -> JsonType:
    """
    Converts a Python object to a representation that can be exported to JSON.

    * Fundamental types (e.g. numeric types) are written as is.
    * Date and time types are serialized in the ISO 8601 format with time zone.
    * A byte array is written as a string with Base64 encoding.
    * UUIDs are written as a UUID string.
    * Enumerations are written as their value.
    * Containers (e.g. `list`, `dict`, `set`, `tuple`) are exported recursively.
    * Objects with properties (including data class types) are converted to a dictionaries of key-value pairs.
    """

    typ: type = type(obj)
    generator = create_serializer(typ)
    return generator.generate(obj)


def json_to_object(typ: TypeLike, data: JsonType, *, context: ModuleType | None = None) -> object:
    """
    Creates an object from a representation that has been de-serialized from JSON.

    When de-serializing a JSON object into a Python object, the following transformations are applied:

    * Fundamental types are parsed as `bool`, `int`, `float` or `str`.
    * Date and time types are parsed from the ISO 8601 format with time zone into the corresponding Python type
      `datetime`, `date` or `time`
    * A byte array is read from a string with Base64 encoding into a `bytes` instance.
    * UUIDs are extracted from a UUID string into a `uuid.UUID` instance.
    * Enumerations are instantiated with a lookup on enumeration value.
    * Containers (e.g. `list`, `dict`, `set`, `tuple`) are parsed recursively.
    * Complex objects with properties (including data class types) are populated from dictionaries of key-value pairs
      using reflection (enumerating type annotations).

    :raises TypeError: A de-serializing engine cannot be constructed for the input type.
    :raises JsonKeyError: Deserialization for a class or union type has failed because a matching member was not found.
    :raises JsonTypeError: Deserialization for data has failed due to a type mismatch.
    """

    # use caller context for evaluating types if no context is supplied
    if context is None:
        this_frame = inspect.currentframe()
        if this_frame is not None:
            caller_frame = this_frame.f_back
            del this_frame

            if caller_frame is not None:
                try:
                    context = sys.modules[caller_frame.f_globals["__name__"]]
                finally:
                    del caller_frame

    parser = create_deserializer(typ, context)
    return parser.parse(data)


def json_dump_string(json_object: JsonType) -> str:
    "Dump an object as a JSON string with a compact representation."

    return json.dumps(json_object, ensure_ascii=False, check_circular=False, separators=(",", ":"))


def json_dump(json_object: JsonType, file: TextIO) -> None:
    json.dump(
        json_object,
        file,
        ensure_ascii=False,
        check_circular=False,
        separators=(",", ":"),
    )
    file.write("\n")
