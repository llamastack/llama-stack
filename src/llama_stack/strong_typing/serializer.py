# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Type-safe data interchange for Python data classes.

:see: https://github.com/hunyadi/strong_typing
"""

import abc
import base64
import datetime
import enum
import functools
import inspect
import ipaddress
import sys
import typing
import uuid
from collections.abc import Callable
from types import FunctionType, MethodType, ModuleType
from typing import (
    Any,
    Generic,
    Literal,
    NamedTuple,
    TypeVar,
    Union,
)

from .core import JsonType
from .exception import JsonTypeError, JsonValueError
from .inspection import (
    TypeLike,
    enum_value_types,
    evaluate_type,
    get_class_properties,
    get_resolved_hints,
    is_dataclass_type,
    is_named_tuple_type,
    is_reserved_property,
    is_type_annotated,
    is_type_enum,
    unwrap_annotated_type,
)
from .mapping import python_field_to_json_property

T = TypeVar("T")


class Serializer(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def generate(self, data: T) -> JsonType: ...


class NoneSerializer(Serializer[None]):
    def generate(self, data: None) -> None:
        # can be directly represented in JSON
        return None


class BoolSerializer(Serializer[bool]):
    def generate(self, data: bool) -> bool:
        # can be directly represented in JSON
        return data


class IntSerializer(Serializer[int]):
    def generate(self, data: int) -> int:
        # can be directly represented in JSON
        return data


class FloatSerializer(Serializer[float]):
    def generate(self, data: float) -> float:
        # can be directly represented in JSON
        return data


class StringSerializer(Serializer[str]):
    def generate(self, data: str) -> str:
        # can be directly represented in JSON
        return data


class BytesSerializer(Serializer[bytes]):
    def generate(self, data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")


class DateTimeSerializer(Serializer[datetime.datetime]):
    def generate(self, obj: datetime.datetime) -> str:
        if obj.tzinfo is None:
            raise JsonValueError(f"timestamp lacks explicit time zone designator: {obj}")
        fmt = obj.isoformat()
        if fmt.endswith("+00:00"):
            fmt = f"{fmt[:-6]}Z"  # Python's isoformat() does not support military time zones like "Zulu" for UTC
        return fmt


class DateSerializer(Serializer[datetime.date]):
    def generate(self, obj: datetime.date) -> str:
        return obj.isoformat()


class TimeSerializer(Serializer[datetime.time]):
    def generate(self, obj: datetime.time) -> str:
        return obj.isoformat()


class UUIDSerializer(Serializer[uuid.UUID]):
    def generate(self, obj: uuid.UUID) -> str:
        return str(obj)


class IPv4Serializer(Serializer[ipaddress.IPv4Address]):
    def generate(self, obj: ipaddress.IPv4Address) -> str:
        return str(obj)


class IPv6Serializer(Serializer[ipaddress.IPv6Address]):
    def generate(self, obj: ipaddress.IPv6Address) -> str:
        return str(obj)


class EnumSerializer(Serializer[enum.Enum]):
    def generate(self, obj: enum.Enum) -> int | str:
        value = obj.value
        if isinstance(value, int):
            return value
        return str(value)


class UntypedListSerializer(Serializer[list]):
    def generate(self, obj: list) -> list[JsonType]:
        return [object_to_json(item) for item in obj]


class UntypedDictSerializer(Serializer[dict]):
    def generate(self, obj: dict) -> dict[str, JsonType]:
        if obj and isinstance(next(iter(obj.keys())), enum.Enum):
            iterator = ((key.value, object_to_json(value)) for key, value in obj.items())
        else:
            iterator = ((str(key), object_to_json(value)) for key, value in obj.items())
        return dict(iterator)


class UntypedSetSerializer(Serializer[set]):
    def generate(self, obj: set) -> list[JsonType]:
        return [object_to_json(item) for item in obj]


class UntypedTupleSerializer(Serializer[tuple]):
    def generate(self, obj: tuple) -> list[JsonType]:
        return [object_to_json(item) for item in obj]


class TypedCollectionSerializer(Serializer, Generic[T]):
    generator: Serializer[T]

    def __init__(self, item_type: type[T], context: ModuleType | None) -> None:
        self.generator = _get_serializer(item_type, context)


class TypedListSerializer(TypedCollectionSerializer[T]):
    def generate(self, obj: list[T]) -> list[JsonType]:
        return [self.generator.generate(item) for item in obj]


class TypedStringDictSerializer(TypedCollectionSerializer[T]):
    def __init__(self, value_type: type[T], context: ModuleType | None) -> None:
        super().__init__(value_type, context)

    def generate(self, obj: dict[str, T]) -> dict[str, JsonType]:
        return {key: self.generator.generate(value) for key, value in obj.items()}


class TypedEnumDictSerializer(TypedCollectionSerializer[T]):
    def __init__(
        self,
        key_type: type[enum.Enum],
        value_type: type[T],
        context: ModuleType | None,
    ) -> None:
        super().__init__(value_type, context)

        value_types = enum_value_types(key_type)
        if len(value_types) != 1:
            raise JsonTypeError(
                f"invalid key type, enumerations must have a consistent member value type but several types found: {value_types}"
            )

        value_type = value_types.pop()
        if value_type is not str:
            raise JsonTypeError("invalid enumeration key type, expected `enum.Enum` with string values")

    def generate(self, obj: dict[enum.Enum, T]) -> dict[str, JsonType]:
        return {key.value: self.generator.generate(value) for key, value in obj.items()}


class TypedSetSerializer(TypedCollectionSerializer[T]):
    def generate(self, obj: set[T]) -> JsonType:
        return [self.generator.generate(item) for item in obj]


class TypedTupleSerializer(Serializer[tuple]):
    item_generators: tuple[Serializer, ...]

    def __init__(self, item_types: tuple[type, ...], context: ModuleType | None) -> None:
        self.item_generators = tuple(_get_serializer(item_type, context) for item_type in item_types)

    def generate(self, obj: tuple) -> list[JsonType]:
        return [item_generator.generate(item) for item_generator, item in zip(self.item_generators, obj, strict=False)]


class CustomSerializer(Serializer):
    converter: Callable[[object], JsonType]

    def __init__(self, converter: Callable[[object], JsonType]) -> None:
        self.converter = converter

    def generate(self, obj: object) -> JsonType:
        return self.converter(obj)


class FieldSerializer(Generic[T]):
    """
    Serializes a Python object field into a JSON property.

    :param field_name: The name of the field in a Python class to read data from.
    :param property_name: The name of the JSON property to write to a JSON `object`.
    :param generator: A compatible serializer that can handle the field's type.
    """

    field_name: str
    property_name: str
    generator: Serializer

    def __init__(self, field_name: str, property_name: str, generator: Serializer[T]) -> None:
        self.field_name = field_name
        self.property_name = property_name
        self.generator = generator

    def generate_field(self, obj: object, object_dict: dict[str, JsonType]) -> None:
        value = getattr(obj, self.field_name)
        if value is not None:
            object_dict[self.property_name] = self.generator.generate(value)


class TypedClassSerializer(Serializer[T]):
    property_generators: list[FieldSerializer]

    def __init__(self, class_type: type[T], context: ModuleType | None) -> None:
        self.property_generators = [
            FieldSerializer(
                field_name,
                python_field_to_json_property(field_name, field_type),
                _get_serializer(field_type, context),
            )
            for field_name, field_type in get_class_properties(class_type)
        ]

    def generate(self, obj: T) -> dict[str, JsonType]:
        object_dict: dict[str, JsonType] = {}
        for property_generator in self.property_generators:
            property_generator.generate_field(obj, object_dict)

        return object_dict


class TypedNamedTupleSerializer(TypedClassSerializer[NamedTuple]):
    def __init__(self, class_type: type[NamedTuple], context: ModuleType | None) -> None:
        super().__init__(class_type, context)


class DataclassSerializer(TypedClassSerializer[T]):
    def __init__(self, class_type: type[T], context: ModuleType | None) -> None:
        super().__init__(class_type, context)


class UnionSerializer(Serializer):
    def generate(self, obj: Any) -> JsonType:
        return object_to_json(obj)


class LiteralSerializer(Serializer):
    generator: Serializer

    def __init__(self, values: tuple[Any, ...], context: ModuleType | None) -> None:
        literal_type_tuple = tuple(type(value) for value in values)
        literal_type_set = set(literal_type_tuple)
        if len(literal_type_set) != 1:
            value_names = ", ".join(repr(value) for value in values)
            raise TypeError(
                f"type `Literal[{value_names}]` expects consistent literal value types but got: {literal_type_tuple}"
            )

        literal_type = literal_type_set.pop()
        self.generator = _get_serializer(literal_type, context)

    def generate(self, obj: Any) -> JsonType:
        return self.generator.generate(obj)


class UntypedNamedTupleSerializer(Serializer):
    fields: dict[str, str]

    def __init__(self, class_type: type[NamedTuple]) -> None:
        # named tuples are also instances of tuple
        self.fields = {}
        field_names: tuple[str, ...] = class_type._fields
        for field_name in field_names:
            self.fields[field_name] = python_field_to_json_property(field_name)

    def generate(self, obj: NamedTuple) -> JsonType:
        object_dict = {}
        for field_name, property_name in self.fields.items():
            value = getattr(obj, field_name)
            object_dict[property_name] = object_to_json(value)

        return object_dict


class UntypedClassSerializer(Serializer):
    def generate(self, obj: object) -> JsonType:
        # iterate over object attributes to get a standard representation
        object_dict = {}
        for name in dir(obj):
            if is_reserved_property(name):
                continue

            value = getattr(obj, name)
            if value is None:
                continue

            # filter instance methods
            if inspect.ismethod(value):
                continue

            object_dict[python_field_to_json_property(name)] = object_to_json(value)

        return object_dict


def create_serializer(typ: TypeLike, context: ModuleType | None = None) -> Serializer:
    """
    Creates a serializer engine to produce an object that can be directly converted into a JSON string.

    When serializing a Python object into a JSON object, the following transformations are applied:

    * Fundamental types (`bool`, `int`, `float` or `str`) are returned as-is.
    * Date and time types (`datetime`, `date` or `time`) produce an ISO 8601 format string with time zone
      (ending with `Z` for UTC).
    * Byte arrays (`bytes`) are written as a string with Base64 encoding.
    * UUIDs (`uuid.UUID`) are written as a UUID string as per RFC 4122.
    * Enumerations yield their enumeration value.
    * Containers (e.g. `list`, `dict`, `set`, `tuple`) are processed recursively.
    * Complex objects with properties (including data class types) generate dictionaries of key-value pairs.

    :raises TypeError: A serializer engine cannot be constructed for the input type.
    """

    if context is None:
        if isinstance(typ, type):
            context = sys.modules[typ.__module__]

    return _get_serializer(typ, context)


def _get_serializer(typ: TypeLike, context: ModuleType | None) -> Serializer:
    if isinstance(typ, (str, typing.ForwardRef)):
        if context is None:
            raise TypeError(f"missing context for evaluating type: {typ}")

        typ = evaluate_type(typ, context)

    if isinstance(typ, type):
        return _fetch_serializer(typ)
    else:
        # special forms are not always hashable
        return _create_serializer(typ, context)


@functools.cache
def _fetch_serializer(typ: type) -> Serializer:
    context = sys.modules[typ.__module__]
    return _create_serializer(typ, context)


def _create_serializer(typ: TypeLike, context: ModuleType | None) -> Serializer:
    # check for well-known types
    if typ is type(None):
        return NoneSerializer()
    elif typ is bool:
        return BoolSerializer()
    elif typ is int:
        return IntSerializer()
    elif typ is float:
        return FloatSerializer()
    elif typ is str:
        return StringSerializer()
    elif typ is bytes:
        return BytesSerializer()
    elif typ is datetime.datetime:
        return DateTimeSerializer()
    elif typ is datetime.date:
        return DateSerializer()
    elif typ is datetime.time:
        return TimeSerializer()
    elif typ is uuid.UUID:
        return UUIDSerializer()
    elif typ is ipaddress.IPv4Address:
        return IPv4Serializer()
    elif typ is ipaddress.IPv6Address:
        return IPv6Serializer()

    # dynamically-typed collection types
    if typ is list:
        return UntypedListSerializer()
    elif typ is dict:
        return UntypedDictSerializer()
    elif typ is set:
        return UntypedSetSerializer()
    elif typ is tuple:
        return UntypedTupleSerializer()

    # generic types (e.g. list, dict, set, etc.)
    origin_type = typing.get_origin(typ)
    if origin_type is list:
        (list_item_type,) = typing.get_args(typ)  # unpack single tuple element
        return TypedListSerializer(list_item_type, context)
    elif origin_type is dict:
        key_type, value_type = typing.get_args(typ)
        if key_type is str:
            return TypedStringDictSerializer(value_type, context)
        elif issubclass(key_type, enum.Enum):
            return TypedEnumDictSerializer(key_type, value_type, context)
    elif origin_type is set:
        (set_member_type,) = typing.get_args(typ)  # unpack single tuple element
        return TypedSetSerializer(set_member_type, context)
    elif origin_type is tuple:
        return TypedTupleSerializer(typing.get_args(typ), context)
    elif origin_type is Union:
        return UnionSerializer()
    elif origin_type is Literal:
        return LiteralSerializer(typing.get_args(typ), context)

    if is_type_annotated(typ):
        return create_serializer(unwrap_annotated_type(typ))

    # check if object has custom serialization method
    convert_func = getattr(typ, "to_json", None)
    if callable(convert_func):
        return CustomSerializer(convert_func)

    if is_type_enum(typ):
        return EnumSerializer()
    if is_dataclass_type(typ):
        return DataclassSerializer(typ, context)
    if is_named_tuple_type(typ):
        if getattr(typ, "__annotations__", None):
            return TypedNamedTupleSerializer(typ, context)
        else:
            return UntypedNamedTupleSerializer(typ)

    # fail early if caller passes an object with an exotic type
    if not isinstance(typ, type) or typ is FunctionType or typ is MethodType or typ is type or typ is ModuleType:
        raise TypeError(f"object of type {typ} cannot be represented in JSON")

    if get_resolved_hints(typ):
        return TypedClassSerializer(typ, context)
    else:
        return UntypedClassSerializer()


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
