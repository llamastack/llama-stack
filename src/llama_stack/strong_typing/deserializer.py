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
import dataclasses
import datetime
import enum
import inspect
import ipaddress
import sys
import typing
import uuid
from collections.abc import Callable
from types import ModuleType
from typing import (
    Any,
    Generic,
    Literal,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
)

from .core import JsonType
from .exception import JsonKeyError, JsonTypeError, JsonValueError
from .inspection import (
    TypeLike,
    create_object,
    enum_value_types,
    evaluate_type,
    get_class_properties,
    get_class_property,
    get_resolved_hints,
    is_dataclass_instance,
    is_dataclass_type,
    is_named_tuple_type,
    is_type_annotated,
    is_type_literal,
    is_type_optional,
    unwrap_annotated_type,
    unwrap_literal_values,
    unwrap_optional_type,
)
from .mapping import python_field_to_json_property
from .name import python_type_to_str

E = TypeVar("E", bound=enum.Enum)
T = TypeVar("T")
R = TypeVar("R")
K = TypeVar("K")
V = TypeVar("V")


class Deserializer(abc.ABC, Generic[T]):
    "Parses a JSON value into a Python type."

    def build(self, context: ModuleType | None) -> None:
        """
        Creates auxiliary parsers that this parser is depending on.

        :param context: A module context for evaluating types specified as a string.
        """

    @abc.abstractmethod
    def parse(self, data: JsonType) -> T:
        """
        Parses a JSON value into a Python type.

        :param data: The JSON value to de-serialize.
        :returns: The Python object that the JSON value de-serializes to.
        """


class NoneDeserializer(Deserializer[None]):
    "Parses JSON `null` values into Python `None`."

    def parse(self, data: JsonType) -> None:
        if data is not None:
            raise JsonTypeError(f"`None` type expects JSON `null` but instead received: {data}")
        return None


class BoolDeserializer(Deserializer[bool]):
    "Parses JSON `boolean` values into Python `bool` type."

    def parse(self, data: JsonType) -> bool:
        if not isinstance(data, bool):
            raise JsonTypeError(f"`bool` type expects JSON `boolean` data but instead received: {data}")
        return bool(data)


class IntDeserializer(Deserializer[int]):
    "Parses JSON `number` values into Python `int` type."

    def parse(self, data: JsonType) -> int:
        if not isinstance(data, int):
            raise JsonTypeError(f"`int` type expects integer data as JSON `number` but instead received: {data}")
        return int(data)


class FloatDeserializer(Deserializer[float]):
    "Parses JSON `number` values into Python `float` type."

    def parse(self, data: JsonType) -> float:
        if not isinstance(data, float) and not isinstance(data, int):
            raise JsonTypeError(f"`int` type expects data as JSON `number` but instead received: {data}")
        return float(data)


class StringDeserializer(Deserializer[str]):
    "Parses JSON `string` values into Python `str` type."

    def parse(self, data: JsonType) -> str:
        if not isinstance(data, str):
            raise JsonTypeError(f"`str` type expects JSON `string` data but instead received: {data}")
        return str(data)


class BytesDeserializer(Deserializer[bytes]):
    "Parses JSON `string` values of Base64-encoded strings into Python `bytes` type."

    def parse(self, data: JsonType) -> bytes:
        if not isinstance(data, str):
            raise JsonTypeError(f"`bytes` type expects JSON `string` data but instead received: {data}")
        return base64.b64decode(data, validate=True)


class DateTimeDeserializer(Deserializer[datetime.datetime]):
    "Parses JSON `string` values representing timestamps in ISO 8601 format to Python `datetime` with time zone."

    def parse(self, data: JsonType) -> datetime.datetime:
        if not isinstance(data, str):
            raise JsonTypeError(f"`datetime` type expects JSON `string` data but instead received: {data}")

        if data.endswith("Z"):
            data = f"{data[:-1]}+00:00"  # Python's isoformat() does not support military time zones like "Zulu" for UTC
        timestamp = datetime.datetime.fromisoformat(data)
        if timestamp.tzinfo is None:
            raise JsonValueError(f"timestamp lacks explicit time zone designator: {data}")
        return timestamp


class DateDeserializer(Deserializer[datetime.date]):
    "Parses JSON `string` values representing dates in ISO 8601 format to Python `date` type."

    def parse(self, data: JsonType) -> datetime.date:
        if not isinstance(data, str):
            raise JsonTypeError(f"`date` type expects JSON `string` data but instead received: {data}")

        return datetime.date.fromisoformat(data)


class TimeDeserializer(Deserializer[datetime.time]):
    "Parses JSON `string` values representing time instances in ISO 8601 format to Python `time` type with time zone."

    def parse(self, data: JsonType) -> datetime.time:
        if not isinstance(data, str):
            raise JsonTypeError(f"`time` type expects JSON `string` data but instead received: {data}")

        return datetime.time.fromisoformat(data)


class UUIDDeserializer(Deserializer[uuid.UUID]):
    "Parses JSON `string` values of UUID strings into Python `uuid.UUID` type."

    def parse(self, data: JsonType) -> uuid.UUID:
        if not isinstance(data, str):
            raise JsonTypeError(f"`UUID` type expects JSON `string` data but instead received: {data}")
        return uuid.UUID(data)


class IPv4Deserializer(Deserializer[ipaddress.IPv4Address]):
    "Parses JSON `string` values of IPv4 address strings into Python `ipaddress.IPv4Address` type."

    def parse(self, data: JsonType) -> ipaddress.IPv4Address:
        if not isinstance(data, str):
            raise JsonTypeError(f"`IPv4Address` type expects JSON `string` data but instead received: {data}")
        return ipaddress.IPv4Address(data)


class IPv6Deserializer(Deserializer[ipaddress.IPv6Address]):
    "Parses JSON `string` values of IPv6 address strings into Python `ipaddress.IPv6Address` type."

    def parse(self, data: JsonType) -> ipaddress.IPv6Address:
        if not isinstance(data, str):
            raise JsonTypeError(f"`IPv6Address` type expects JSON `string` data but instead received: {data}")
        return ipaddress.IPv6Address(data)


class ListDeserializer(Deserializer[list[T]]):
    "Recursively de-serializes a JSON array into a Python `list`."

    item_type: type[T]
    item_parser: Deserializer

    def __init__(self, item_type: type[T]) -> None:
        self.item_type = item_type

    def build(self, context: ModuleType | None) -> None:
        self.item_parser = _get_deserializer(self.item_type, context)

    def parse(self, data: JsonType) -> list[T]:
        if not isinstance(data, list):
            type_name = python_type_to_str(self.item_type)
            raise JsonTypeError(f"type `List[{type_name}]` expects JSON `array` data but instead received: {data}")

        return [self.item_parser.parse(item) for item in data]


class DictDeserializer(Deserializer[dict[K, V]]):
    "Recursively de-serializes a JSON object into a Python `dict`."

    key_type: type[K]
    value_type: type[V]
    value_parser: Deserializer[V]

    def __init__(self, key_type: type[K], value_type: type[V]) -> None:
        self.key_type = key_type
        self.value_type = value_type
        self._check_key_type()

    def build(self, context: ModuleType | None) -> None:
        self.value_parser = _get_deserializer(self.value_type, context)

    def _check_key_type(self) -> None:
        if self.key_type is str:
            return

        if issubclass(self.key_type, enum.Enum):
            value_types = enum_value_types(self.key_type)
            if len(value_types) != 1:
                raise JsonTypeError(
                    f"type `{self.container_type}` has invalid key type, "
                    f"enumerations must have a consistent member value type but several types found: {value_types}"
                )
            value_type = value_types.pop()
            if value_type is not str:
                f"`type `{self.container_type}` has invalid enumeration key type, expected `enum.Enum` with string values"
            return

        raise JsonTypeError(
            f"`type `{self.container_type}` has invalid key type, expected `str` or `enum.Enum` with string values"
        )

    @property
    def container_type(self) -> str:
        key_type_name = python_type_to_str(self.key_type)
        value_type_name = python_type_to_str(self.value_type)
        return f"Dict[{key_type_name}, {value_type_name}]"

    def parse(self, data: JsonType) -> dict[K, V]:
        if not isinstance(data, dict):
            raise JsonTypeError(
                f"`type `{self.container_type}` expects JSON `object` data but instead received: {data}"
            )

        return dict(
            (self.key_type(key), self.value_parser.parse(value))  # type: ignore[call-arg]
            for key, value in data.items()
        )


class SetDeserializer(Deserializer[set[T]]):
    "Recursively de-serializes a JSON list into a Python `set`."

    member_type: type[T]
    member_parser: Deserializer

    def __init__(self, member_type: type[T]) -> None:
        self.member_type = member_type

    def build(self, context: ModuleType | None) -> None:
        self.member_parser = _get_deserializer(self.member_type, context)

    def parse(self, data: JsonType) -> set[T]:
        if not isinstance(data, list):
            type_name = python_type_to_str(self.member_type)
            raise JsonTypeError(f"type `Set[{type_name}]` expects JSON `array` data but instead received: {data}")

        return set(self.member_parser.parse(item) for item in data)


class TupleDeserializer(Deserializer[tuple[Any, ...]]):
    "Recursively de-serializes a JSON list into a Python `tuple`."

    item_types: tuple[type[Any], ...]
    item_parsers: tuple[Deserializer[Any], ...]

    def __init__(self, item_types: tuple[type[Any], ...]) -> None:
        self.item_types = item_types

    def build(self, context: ModuleType | None) -> None:
        self.item_parsers = tuple(_get_deserializer(item_type, context) for item_type in self.item_types)

    @property
    def container_type(self) -> str:
        type_names = ", ".join(python_type_to_str(item_type) for item_type in self.item_types)
        return f"Tuple[{type_names}]"

    def parse(self, data: JsonType) -> tuple[Any, ...]:
        if not isinstance(data, list) or len(data) != len(self.item_parsers):
            if not isinstance(data, list):
                raise JsonTypeError(
                    f"type `{self.container_type}` expects JSON `array` data but instead received: {data}"
                )
            else:
                count = len(self.item_parsers)
                raise JsonValueError(
                    f"type `{self.container_type}` expects a JSON `array` of length {count} but received length {len(data)}"
                )

        return tuple(item_parser.parse(item) for item_parser, item in zip(self.item_parsers, data, strict=False))


class UnionDeserializer(Deserializer):
    "De-serializes a JSON value (of any type) into a Python union type."

    member_types: tuple[type, ...]
    member_parsers: tuple[Deserializer, ...]

    def __init__(self, member_types: tuple[type, ...]) -> None:
        self.member_types = member_types

    def build(self, context: ModuleType | None) -> None:
        self.member_parsers = tuple(_get_deserializer(member_type, context) for member_type in self.member_types)

    def parse(self, data: JsonType) -> Any:
        for member_parser in self.member_parsers:
            # iterate over potential types of discriminated union
            try:
                return member_parser.parse(data)
            except (JsonKeyError, JsonTypeError):
                # indicates a required field is missing from JSON dict -OR- the data cannot be cast to the expected type,
                # i.e. we don't have the type that we are looking for
                continue

        type_names = ", ".join(python_type_to_str(member_type) for member_type in self.member_types)
        raise JsonKeyError(f"type `Union[{type_names}]` could not be instantiated from: {data}")


def get_literal_properties(typ: type) -> set[str]:
    "Returns the names of all properties in a class that are of a literal type."

    return set(
        property_name for property_name, property_type in get_class_properties(typ) if is_type_literal(property_type)
    )


def get_discriminating_properties(types: tuple[type, ...]) -> set[str]:
    "Returns a set of properties with literal type that are common across all specified classes."

    if not types or not all(isinstance(typ, type) for typ in types):
        return set()

    props = get_literal_properties(types[0])
    for typ in types[1:]:
        props = props & get_literal_properties(typ)

    return props


class TaggedUnionDeserializer(Deserializer):
    "De-serializes a JSON value with one or more disambiguating properties into a Python union type."

    member_types: tuple[type, ...]
    disambiguating_properties: set[str]
    member_parsers: dict[tuple[str, Any], Deserializer]

    def __init__(self, member_types: tuple[type, ...]) -> None:
        self.member_types = member_types
        self.disambiguating_properties = get_discriminating_properties(member_types)

    def build(self, context: ModuleType | None) -> None:
        self.member_parsers = {}
        for member_type in self.member_types:
            for property_name in self.disambiguating_properties:
                literal_type = get_class_property(member_type, property_name)
                if not literal_type:
                    continue

                for literal_value in unwrap_literal_values(literal_type):
                    tpl = (property_name, literal_value)
                    if tpl in self.member_parsers:
                        raise JsonTypeError(
                            f"disambiguating property `{property_name}` in type `{self.union_type}` has a duplicate value: {literal_value}"
                        )

                    self.member_parsers[tpl] = _get_deserializer(member_type, context)

    @property
    def union_type(self) -> str:
        type_names = ", ".join(python_type_to_str(member_type) for member_type in self.member_types)
        return f"Union[{type_names}]"

    def parse(self, data: JsonType) -> Any:
        if not isinstance(data, dict):
            raise JsonTypeError(
                f"tagged union type `{self.union_type}` expects JSON `object` data but instead received: {data}"
            )

        for property_name in self.disambiguating_properties:
            disambiguating_value = data.get(property_name)
            if disambiguating_value is None:
                continue

            member_parser = self.member_parsers.get((property_name, disambiguating_value))
            if member_parser is None:
                raise JsonTypeError(
                    f"disambiguating property value is invalid for tagged union type `{self.union_type}`: {data}"
                )

            return member_parser.parse(data)

        raise JsonTypeError(
            f"disambiguating property value is missing for tagged union type `{self.union_type}`: {data}"
        )


class LiteralDeserializer(Deserializer):
    "De-serializes a JSON value into a Python literal type."

    values: tuple[Any, ...]
    parser: Deserializer

    def __init__(self, values: tuple[Any, ...]) -> None:
        self.values = values

    def build(self, context: ModuleType | None) -> None:
        literal_type_tuple = tuple(type(value) for value in self.values)
        literal_type_set = set(literal_type_tuple)
        if len(literal_type_set) != 1:
            value_names = ", ".join(repr(value) for value in self.values)
            raise TypeError(
                f"type `Literal[{value_names}]` expects consistent literal value types but got: {literal_type_tuple}"
            )

        literal_type = literal_type_set.pop()
        self.parser = _get_deserializer(literal_type, context)

    def parse(self, data: JsonType) -> Any:
        value = self.parser.parse(data)
        if value not in self.values:
            value_names = ", ".join(repr(value) for value in self.values)
            raise JsonTypeError(f"type `Literal[{value_names}]` could not be instantiated from: {data}")
        return value


class EnumDeserializer(Deserializer[E]):
    "Returns an enumeration instance based on the enumeration value read from a JSON value."

    enum_type: type[E]

    def __init__(self, enum_type: type[E]) -> None:
        self.enum_type = enum_type

    def parse(self, data: JsonType) -> E:
        return self.enum_type(data)


class CustomDeserializer(Deserializer[T]):
    "Uses the `from_json` class method in class to de-serialize the object from JSON."

    converter: Callable[[JsonType], T]

    def __init__(self, converter: Callable[[JsonType], T]) -> None:
        self.converter = converter

    def parse(self, data: JsonType) -> T:
        return self.converter(data)


class FieldDeserializer(abc.ABC, Generic[T, R]):
    """
    Deserializes a JSON property into a Python object field.

    :param property_name: The name of the JSON property to read from a JSON `object`.
    :param field_name: The name of the field in a Python class to write data to.
    :param parser: A compatible deserializer that can handle the field's type.
    """

    property_name: str
    field_name: str
    parser: Deserializer[T]

    def __init__(self, property_name: str, field_name: str, parser: Deserializer[T]) -> None:
        self.property_name = property_name
        self.field_name = field_name
        self.parser = parser

    @abc.abstractmethod
    def parse_field(self, data: dict[str, JsonType]) -> R: ...


class RequiredFieldDeserializer(FieldDeserializer[T, T]):
    "Deserializes a JSON property into a mandatory Python object field."

    def parse_field(self, data: dict[str, JsonType]) -> T:
        if self.property_name not in data:
            raise JsonKeyError(f"missing required property `{self.property_name}` from JSON object: {data}")

        return self.parser.parse(data[self.property_name])


class OptionalFieldDeserializer(FieldDeserializer[T, Optional[T]]):
    "Deserializes a JSON property into an optional Python object field with a default value of `None`."

    def parse_field(self, data: dict[str, JsonType]) -> T | None:
        value = data.get(self.property_name)
        if value is not None:
            return self.parser.parse(value)
        else:
            return None


class DefaultFieldDeserializer(FieldDeserializer[T, T]):
    "Deserializes a JSON property into a Python object field with an explicit default value."

    default_value: T

    def __init__(
        self,
        property_name: str,
        field_name: str,
        parser: Deserializer,
        default_value: T,
    ) -> None:
        super().__init__(property_name, field_name, parser)
        self.default_value = default_value

    def parse_field(self, data: dict[str, JsonType]) -> T:
        value = data.get(self.property_name)
        if value is not None:
            return self.parser.parse(value)
        else:
            return self.default_value


class DefaultFactoryFieldDeserializer(FieldDeserializer[T, T]):
    "Deserializes a JSON property into an optional Python object field with an explicit default value factory."

    default_factory: Callable[[], T]

    def __init__(
        self,
        property_name: str,
        field_name: str,
        parser: Deserializer[T],
        default_factory: Callable[[], T],
    ) -> None:
        super().__init__(property_name, field_name, parser)
        self.default_factory = default_factory

    def parse_field(self, data: dict[str, JsonType]) -> T:
        value = data.get(self.property_name)
        if value is not None:
            return self.parser.parse(value)
        else:
            return self.default_factory()


class ClassDeserializer(Deserializer[T]):
    "Base class for de-serializing class-like types such as data classes, named tuples and regular classes."

    class_type: type
    property_parsers: list[FieldDeserializer]
    property_fields: set[str]

    def __init__(self, class_type: type[T]) -> None:
        self.class_type = class_type

    def assign(self, property_parsers: list[FieldDeserializer]) -> None:
        self.property_parsers = property_parsers
        self.property_fields = set(property_parser.property_name for property_parser in property_parsers)

    def parse(self, data: JsonType) -> T:
        if not isinstance(data, dict):
            type_name = python_type_to_str(self.class_type)
            raise JsonTypeError(f"`type `{type_name}` expects JSON `object` data but instead received: {data}")

        object_data: dict[str, JsonType] = typing.cast(dict[str, JsonType], data)

        field_values = {}
        for property_parser in self.property_parsers:
            field_values[property_parser.field_name] = property_parser.parse_field(object_data)

        if not self.property_fields.issuperset(object_data):
            unassigned_names = [name for name in object_data if name not in self.property_fields]
            raise JsonKeyError(f"unrecognized fields in JSON object: {unassigned_names}")

        return self.create(**field_values)

    def create(self, **field_values: Any) -> T:
        "Instantiates an object with a collection of property values."

        obj: T = create_object(self.class_type)

        # use `setattr` on newly created object instance
        for field_name, field_value in field_values.items():
            setattr(obj, field_name, field_value)
        return obj


class NamedTupleDeserializer(ClassDeserializer[NamedTuple]):
    "De-serializes a named tuple from a JSON `object`."

    def build(self, context: ModuleType | None) -> None:
        property_parsers: list[FieldDeserializer] = [
            RequiredFieldDeserializer(field_name, field_name, _get_deserializer(field_type, context))
            for field_name, field_type in get_resolved_hints(self.class_type).items()
        ]
        super().assign(property_parsers)

    def create(self, **field_values: Any) -> NamedTuple:
        # mypy fails to deduce that this class returns NamedTuples only, hence the `ignore` directive
        return self.class_type(**field_values)  # type: ignore[no-any-return]


class DataclassDeserializer(ClassDeserializer[T]):
    "De-serializes a data class from a JSON `object`."

    def __init__(self, class_type: type[T]) -> None:
        if not dataclasses.is_dataclass(class_type):
            raise TypeError("expected: data-class type")
        super().__init__(class_type)  # type: ignore[arg-type]

    def build(self, context: ModuleType | None) -> None:
        property_parsers: list[FieldDeserializer] = []
        resolved_hints = get_resolved_hints(self.class_type)
        for field in dataclasses.fields(self.class_type):
            field_type = resolved_hints[field.name]
            property_name = python_field_to_json_property(field.name, field_type)

            is_optional = is_type_optional(field_type)
            has_default = field.default is not dataclasses.MISSING
            has_default_factory = field.default_factory is not dataclasses.MISSING

            if is_optional:
                required_type: type[T] = unwrap_optional_type(field_type)
            else:
                required_type = field_type

            parser = _get_deserializer(required_type, context)

            if has_default:
                field_parser: FieldDeserializer = DefaultFieldDeserializer(
                    property_name, field.name, parser, field.default
                )
            elif has_default_factory:
                default_factory = typing.cast(Callable[[], Any], field.default_factory)
                field_parser = DefaultFactoryFieldDeserializer(property_name, field.name, parser, default_factory)
            elif is_optional:
                field_parser = OptionalFieldDeserializer(property_name, field.name, parser)
            else:
                field_parser = RequiredFieldDeserializer(property_name, field.name, parser)

            property_parsers.append(field_parser)

        super().assign(property_parsers)


class FrozenDataclassDeserializer(DataclassDeserializer[T]):
    "De-serializes a frozen data class from a JSON `object`."

    def create(self, **field_values: Any) -> T:
        "Instantiates an object with a collection of property values."

        # create object instance without calling `__init__`
        obj: T = create_object(self.class_type)

        # can't use `setattr` on frozen dataclasses, pass member variable values to `__init__`
        obj.__init__(**field_values)  # type: ignore
        return obj


class TypedClassDeserializer(ClassDeserializer[T]):
    "De-serializes a class with type annotations from a JSON `object` by iterating over class properties."

    def build(self, context: ModuleType | None) -> None:
        property_parsers: list[FieldDeserializer] = []
        for field_name, field_type in get_resolved_hints(self.class_type).items():
            property_name = python_field_to_json_property(field_name, field_type)

            is_optional = is_type_optional(field_type)

            if is_optional:
                required_type: type[T] = unwrap_optional_type(field_type)
            else:
                required_type = field_type

            parser = _get_deserializer(required_type, context)

            if is_optional:
                field_parser: FieldDeserializer = OptionalFieldDeserializer(property_name, field_name, parser)
            else:
                field_parser = RequiredFieldDeserializer(property_name, field_name, parser)

            property_parsers.append(field_parser)

        super().assign(property_parsers)


def create_deserializer(typ: TypeLike, context: ModuleType | None = None) -> Deserializer:
    """
    Creates a de-serializer engine to produce a Python object from an object obtained from a JSON string.

    When de-serializing a JSON object into a Python object, the following transformations are applied:

    * Fundamental types are parsed as `bool`, `int`, `float` or `str`.
    * Date and time types are parsed from the ISO 8601 format with time zone into the corresponding Python type
      `datetime`, `date` or `time`.
    * Byte arrays are read from a string with Base64 encoding into a `bytes` instance.
    * UUIDs are extracted from a UUID string compliant with RFC 4122 into a `uuid.UUID` instance.
    * Enumerations are instantiated with a lookup on enumeration value.
    * Containers (e.g. `list`, `dict`, `set`, `tuple`) are parsed recursively.
    * Complex objects with properties (including data class types) are populated from dictionaries of key-value pairs
      using reflection (enumerating type annotations).

    :raises TypeError: A de-serializer engine cannot be constructed for the input type.
    """

    if context is None:
        if isinstance(typ, type):
            context = sys.modules[typ.__module__]

    return _get_deserializer(typ, context)


_CACHE: dict[tuple[str, str], Deserializer] = {}


def _get_deserializer(typ: TypeLike, context: ModuleType | None) -> Deserializer:
    "Creates or re-uses a de-serializer engine to parse an object obtained from a JSON string."

    cache_key = None

    if isinstance(typ, (str, typing.ForwardRef)):
        if context is None:
            raise TypeError(f"missing context for evaluating type: {typ}")

        if isinstance(typ, str):
            if hasattr(context, typ):
                cache_key = (context.__name__, typ)
        elif isinstance(typ, typing.ForwardRef):
            if hasattr(context, typ.__forward_arg__):
                cache_key = (context.__name__, typ.__forward_arg__)

        typ = evaluate_type(typ, context)

    typ = unwrap_annotated_type(typ) if is_type_annotated(typ) else typ

    if isinstance(typ, type) and typing.get_origin(typ) is None:
        cache_key = (typ.__module__, typ.__name__)

    if cache_key is not None:
        deserializer = _CACHE.get(cache_key)
        if deserializer is None:
            deserializer = _create_deserializer(typ)

            # store de-serializer immediately in cache to avoid stack overflow for recursive types
            _CACHE[cache_key] = deserializer

            if isinstance(typ, type):
                # use type's own module as context for evaluating member types
                context = sys.modules[typ.__module__]

            # create any de-serializers this de-serializer is depending on
            deserializer.build(context)
    else:
        # special forms are not always hashable, create a new de-serializer every time
        deserializer = _create_deserializer(typ)
        deserializer.build(context)

    return deserializer


def _create_deserializer(typ: TypeLike) -> Deserializer:
    "Creates a de-serializer engine to parse an object obtained from a JSON string."

    # check for well-known types
    if typ is type(None):
        return NoneDeserializer()
    elif typ is bool:
        return BoolDeserializer()
    elif typ is int:
        return IntDeserializer()
    elif typ is float:
        return FloatDeserializer()
    elif typ is str:
        return StringDeserializer()
    elif typ is bytes:
        return BytesDeserializer()
    elif typ is datetime.datetime:
        return DateTimeDeserializer()
    elif typ is datetime.date:
        return DateDeserializer()
    elif typ is datetime.time:
        return TimeDeserializer()
    elif typ is uuid.UUID:
        return UUIDDeserializer()
    elif typ is ipaddress.IPv4Address:
        return IPv4Deserializer()
    elif typ is ipaddress.IPv6Address:
        return IPv6Deserializer()

    # dynamically-typed collection types
    if typ is list:
        raise TypeError("explicit item type required: use `List[T]` instead of `list`")
    if typ is dict:
        raise TypeError("explicit key and value types required: use `Dict[K, V]` instead of `dict`")
    if typ is set:
        raise TypeError("explicit member type required: use `Set[T]` instead of `set`")
    if typ is tuple:
        raise TypeError("explicit item type list required: use `Tuple[T, ...]` instead of `tuple`")

    # generic types (e.g. list, dict, set, etc.)
    origin_type = typing.get_origin(typ)
    if origin_type is list:
        (list_item_type,) = typing.get_args(typ)  # unpack single tuple element
        return ListDeserializer(list_item_type)
    elif origin_type is dict:
        key_type, value_type = typing.get_args(typ)
        return DictDeserializer(key_type, value_type)
    elif origin_type is set:
        (set_member_type,) = typing.get_args(typ)  # unpack single tuple element
        return SetDeserializer(set_member_type)
    elif origin_type is tuple:
        return TupleDeserializer(typing.get_args(typ))
    elif origin_type is Union:
        union_args = typing.get_args(typ)
        if get_discriminating_properties(union_args):
            return TaggedUnionDeserializer(union_args)
        else:
            return UnionDeserializer(union_args)
    elif origin_type is Literal:
        return LiteralDeserializer(typing.get_args(typ))

    if not inspect.isclass(typ):
        if is_dataclass_instance(typ):
            raise TypeError(f"dataclass type expected but got instance: {typ}")
        else:
            raise TypeError(f"unable to de-serialize unrecognized type: {typ}")

    if issubclass(typ, enum.Enum):
        return EnumDeserializer(typ)

    if is_named_tuple_type(typ):
        return NamedTupleDeserializer(typ)

    # check if object has custom serialization method
    convert_func = getattr(typ, "from_json", None)
    if callable(convert_func):
        return CustomDeserializer(convert_func)

    if is_dataclass_type(typ):
        dataclass_params = getattr(typ, "__dataclass_params__", None)
        if dataclass_params is not None and dataclass_params.frozen:
            return FrozenDataclassDeserializer(typ)
        else:
            return DataclassDeserializer(typ)

    return TypedClassDeserializer(typ)
