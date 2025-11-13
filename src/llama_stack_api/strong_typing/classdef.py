# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import copy
import dataclasses
import datetime
import decimal
import enum
import ipaddress
import math
import re
import sys
import types
import typing
import uuid
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, Union

from .auxiliary import (
    Alias,
    Annotated,
    MaxLength,
    Precision,
    float32,
    float64,
    int16,
    int32,
    int64,
)
from .core import JsonType, Schema
from .docstring import Docstring, DocstringParam
from .inspection import TypeLike
from .serialization import json_to_object, object_to_json

T = TypeVar("T")


@dataclass
class JsonSchemaNode:
    title: str | None
    description: str | None


@dataclass
class JsonSchemaType(JsonSchemaNode):
    type: str
    format: str | None


@dataclass
class JsonSchemaBoolean(JsonSchemaType):
    type: Literal["boolean"]
    const: bool | None
    default: bool | None
    examples: list[bool] | None


@dataclass
class JsonSchemaInteger(JsonSchemaType):
    type: Literal["integer"]
    const: int | None
    default: int | None
    examples: list[int] | None
    enum: list[int] | None
    minimum: int | None
    maximum: int | None


@dataclass
class JsonSchemaNumber(JsonSchemaType):
    type: Literal["number"]
    const: float | None
    default: float | None
    examples: list[float] | None
    minimum: float | None
    maximum: float | None
    exclusiveMinimum: float | None
    exclusiveMaximum: float | None
    multipleOf: float | None


@dataclass
class JsonSchemaString(JsonSchemaType):
    type: Literal["string"]
    const: str | None
    default: str | None
    examples: list[str] | None
    enum: list[str] | None
    minLength: int | None
    maxLength: int | None


@dataclass
class JsonSchemaArray(JsonSchemaType):
    type: Literal["array"]
    items: "JsonSchemaAny"


@dataclass
class JsonSchemaObject(JsonSchemaType):
    type: Literal["object"]
    properties: dict[str, "JsonSchemaAny"] | None
    additionalProperties: bool | None
    required: list[str] | None


@dataclass
class JsonSchemaRef(JsonSchemaNode):
    ref: Annotated[str, Alias("$ref")]


@dataclass
class JsonSchemaAllOf(JsonSchemaNode):
    allOf: list["JsonSchemaAny"]


@dataclass
class JsonSchemaAnyOf(JsonSchemaNode):
    anyOf: list["JsonSchemaAny"]


@dataclass
class Discriminator:
    propertyName: str
    mapping: dict[str, str]


@dataclass
class JsonSchemaOneOf(JsonSchemaNode):
    oneOf: list["JsonSchemaAny"]
    discriminator: Discriminator | None


JsonSchemaAny = Union[
    JsonSchemaRef,
    JsonSchemaBoolean,
    JsonSchemaInteger,
    JsonSchemaNumber,
    JsonSchemaString,
    JsonSchemaArray,
    JsonSchemaObject,
    JsonSchemaOneOf,
]


@dataclass
class JsonSchemaTopLevelObject(JsonSchemaObject):
    schema: Annotated[str, Alias("$schema")]
    definitions: dict[str, JsonSchemaAny] | None


def integer_range_to_type(min_value: float, max_value: float) -> type:
    if min_value >= -(2**15) and max_value < 2**15:
        return int16
    elif min_value >= -(2**31) and max_value < 2**31:
        return int32
    else:
        return int64


def enum_safe_name(name: str) -> str:
    name = re.sub(r"\W", "_", name)
    is_dunder = name.startswith("__")
    is_sunder = name.startswith("_") and name.endswith("_")
    if is_dunder or is_sunder:  # provide an alternative for dunder and sunder names
        name = f"v{name}"
    return name


def enum_values_to_type(
    module: types.ModuleType,
    name: str,
    values: dict[str, Any],
    title: str | None = None,
    description: str | None = None,
) -> type[enum.Enum]:
    enum_class: type[enum.Enum] = enum.Enum(name, values)  # type: ignore

    # assign the newly created type to the same module where the defining class is
    enum_class.__module__ = module.__name__
    enum_class.__doc__ = str(Docstring(short_description=title, long_description=description))
    setattr(module, name, enum_class)

    return enum.unique(enum_class)


def schema_to_type(schema: Schema, *, module: types.ModuleType, class_name: str) -> TypeLike:
    """
    Creates a Python type from a JSON schema.

    :param schema: The JSON schema that the types would correspond to.
    :param module: The module in which to create the new types.
    :param class_name: The name assigned to the top-level class.
    """

    top_node = typing.cast(JsonSchemaTopLevelObject, json_to_object(JsonSchemaTopLevelObject, schema))
    if top_node.definitions is not None:
        for type_name, type_node in top_node.definitions.items():
            type_def = node_to_typedef(module, type_name, type_node)
            if type_def.default is not dataclasses.MISSING:
                raise TypeError("disallowed: `default` for top-level type definitions")

            type_def.type.__module__ = module.__name__
            setattr(module, type_name, type_def.type)

    return node_to_typedef(module, class_name, top_node).type


@dataclass
class TypeDef:
    type: TypeLike
    default: Any = dataclasses.MISSING


def json_to_value(target_type: TypeLike, data: JsonType) -> Any:
    if data is not None:
        return json_to_object(target_type, data)
    else:
        return dataclasses.MISSING


def node_to_typedef(module: types.ModuleType, context: str, node: JsonSchemaNode) -> TypeDef:
    if isinstance(node, JsonSchemaRef):
        match_obj = re.match(r"^#/definitions/(\w+)$", node.ref)
        if not match_obj:
            raise ValueError(f"invalid reference: {node.ref}")

        type_name = match_obj.group(1)
        return TypeDef(getattr(module, type_name), dataclasses.MISSING)

    elif isinstance(node, JsonSchemaBoolean):
        if node.const is not None:
            return TypeDef(Literal[node.const], dataclasses.MISSING)

        default = json_to_value(bool, node.default)
        return TypeDef(bool, default)

    elif isinstance(node, JsonSchemaInteger):
        if node.const is not None:
            return TypeDef(Literal[node.const], dataclasses.MISSING)

        integer_type: TypeLike
        if node.format == "int16":
            integer_type = int16
        elif node.format == "int32":
            integer_type = int32
        elif node.format == "int64":
            integer_type = int64
        else:
            if node.enum is not None:
                integer_type = integer_range_to_type(min(node.enum), max(node.enum))
            elif node.minimum is not None and node.maximum is not None:
                integer_type = integer_range_to_type(node.minimum, node.maximum)
            else:
                integer_type = int

        default = json_to_value(integer_type, node.default)
        return TypeDef(integer_type, default)

    elif isinstance(node, JsonSchemaNumber):
        if node.const is not None:
            return TypeDef(Literal[node.const], dataclasses.MISSING)

        number_type: TypeLike
        if node.format == "float32":
            number_type = float32
        elif node.format == "float64":
            number_type = float64
        else:
            if (
                node.exclusiveMinimum is not None
                and node.exclusiveMaximum is not None
                and node.exclusiveMinimum == -node.exclusiveMaximum
            ):
                integer_digits = round(math.log10(node.exclusiveMaximum))
            else:
                integer_digits = None

            if node.multipleOf is not None:
                decimal_digits = -round(math.log10(node.multipleOf))
            else:
                decimal_digits = None

            if integer_digits is not None and decimal_digits is not None:
                number_type = Annotated[
                    decimal.Decimal,
                    Precision(integer_digits + decimal_digits, decimal_digits),
                ]
            else:
                number_type = float

        default = json_to_value(number_type, node.default)
        return TypeDef(number_type, default)

    elif isinstance(node, JsonSchemaString):
        if node.const is not None:
            return TypeDef(Literal[node.const], dataclasses.MISSING)

        string_type: TypeLike
        if node.format == "date-time":
            string_type = datetime.datetime
        elif node.format == "uuid":
            string_type = uuid.UUID
        elif node.format == "ipv4":
            string_type = ipaddress.IPv4Address
        elif node.format == "ipv6":
            string_type = ipaddress.IPv6Address

        elif node.enum is not None:
            string_type = enum_values_to_type(
                module,
                context,
                {enum_safe_name(e): e for e in node.enum},
                title=node.title,
                description=node.description,
            )

        elif node.maxLength is not None:
            string_type = Annotated[str, MaxLength(node.maxLength)]
        else:
            string_type = str

        default = json_to_value(string_type, node.default)
        return TypeDef(string_type, default)

    elif isinstance(node, JsonSchemaArray):
        type_def = node_to_typedef(module, context, node.items)
        if type_def.default is not dataclasses.MISSING:
            raise TypeError("disallowed: `default` for array element type")
        list_type = list[(type_def.type,)]  # type: ignore
        return TypeDef(list_type, dataclasses.MISSING)

    elif isinstance(node, JsonSchemaObject):
        if node.properties is None:
            return TypeDef(JsonType, dataclasses.MISSING)

        if node.additionalProperties is None or node.additionalProperties is not False:
            raise TypeError("expected: `additionalProperties` equals `false`")

        required = node.required if node.required is not None else []

        class_name = context

        fields: list[tuple[str, Any, dataclasses.Field]] = []
        params: dict[str, DocstringParam] = {}
        for prop_name, prop_node in node.properties.items():
            type_def = node_to_typedef(module, f"{class_name}__{prop_name}", prop_node)
            if prop_name in required:
                prop_type = type_def.type
            else:
                prop_type = Union[(None, type_def.type)]
            fields.append((prop_name, prop_type, dataclasses.field(default=type_def.default)))
            prop_desc = prop_node.title or prop_node.description
            if prop_desc is not None:
                params[prop_name] = DocstringParam(prop_name, prop_desc)

        fields.sort(key=lambda t: t[2].default is not dataclasses.MISSING)
        if sys.version_info >= (3, 12):
            class_type = dataclasses.make_dataclass(class_name, fields, module=module.__name__)
        else:
            class_type = dataclasses.make_dataclass(class_name, fields, namespace={"__module__": module.__name__})
        class_type.__doc__ = str(
            Docstring(
                short_description=node.title,
                long_description=node.description,
                params=params,
            )
        )
        setattr(module, class_name, class_type)
        return TypeDef(class_type, dataclasses.MISSING)

    elif isinstance(node, JsonSchemaOneOf):
        union_defs = tuple(node_to_typedef(module, context, n) for n in node.oneOf)
        if any(d.default is not dataclasses.MISSING for d in union_defs):
            raise TypeError("disallowed: `default` for union member type")
        union_types = tuple(d.type for d in union_defs)
        return TypeDef(Union[union_types], dataclasses.MISSING)

    raise NotImplementedError()


@dataclass
class SchemaFlatteningOptions:
    qualified_names: bool = False
    recursive: bool = False


def flatten_schema(schema: Schema, *, options: SchemaFlatteningOptions | None = None) -> Schema:
    top_node = typing.cast(JsonSchemaTopLevelObject, json_to_object(JsonSchemaTopLevelObject, schema))
    flattener = SchemaFlattener(options)
    obj = flattener.flatten(top_node)
    return typing.cast(Schema, object_to_json(obj))


class SchemaFlattener:
    options: SchemaFlatteningOptions

    def __init__(self, options: SchemaFlatteningOptions | None = None) -> None:
        self.options = options or SchemaFlatteningOptions()

    def flatten(self, source_node: JsonSchemaObject) -> JsonSchemaObject:
        if source_node.type != "object":
            return source_node

        source_props = source_node.properties or {}
        target_props: dict[str, JsonSchemaAny] = {}

        source_reqs = source_node.required or []
        target_reqs: list[str] = []

        for name, prop in source_props.items():
            if not isinstance(prop, JsonSchemaObject):
                target_props[name] = prop
                if name in source_reqs:
                    target_reqs.append(name)
                continue

            if self.options.recursive:
                obj = self.flatten(prop)
            else:
                obj = prop
            if obj.properties is not None:
                if self.options.qualified_names:
                    target_props.update((f"{name}.{n}", p) for n, p in obj.properties.items())
                else:
                    target_props.update(obj.properties.items())
            if obj.required is not None:
                if self.options.qualified_names:
                    target_reqs.extend(f"{name}.{n}" for n in obj.required)
                else:
                    target_reqs.extend(obj.required)

        target_node = copy.copy(source_node)
        target_node.properties = target_props or None
        target_node.additionalProperties = False
        target_node.required = target_reqs or None
        return target_node
