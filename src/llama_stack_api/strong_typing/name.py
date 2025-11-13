# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Type-safe data interchange for Python data classes.

:see: https://github.com/hunyadi/strong_typing
"""

import typing
from typing import Any, Literal, Union

from .auxiliary import _auxiliary_types
from .inspection import (
    TypeLike,
    is_generic_dict,
    is_generic_list,
    is_generic_sequence,
    is_type_optional,
    is_type_union,
    unwrap_generic_dict,
    unwrap_generic_list,
    unwrap_generic_sequence,
    unwrap_optional_type,
    unwrap_union_types,
)


class TypeFormatter:
    """
    Type formatter.

    :param use_union_operator: Whether to emit union types as `X | Y` as per PEP 604.
    """

    use_union_operator: bool

    def __init__(self, use_union_operator: bool = False) -> None:
        self.use_union_operator = use_union_operator

    def union_to_str(self, data_type_args: tuple[TypeLike, ...]) -> str:
        if self.use_union_operator:
            return " | ".join(self.python_type_to_str(t) for t in data_type_args)
        else:
            if len(data_type_args) == 2 and type(None) in data_type_args:
                # Optional[T] is represented as Union[T, None]
                origin_name = "Optional"
                data_type_args = tuple(t for t in data_type_args if t is not type(None))
            else:
                origin_name = "Union"

            args = ", ".join(self.python_type_to_str(t) for t in data_type_args)
            return f"{origin_name}[{args}]"

    def plain_type_to_str(self, data_type: TypeLike) -> str:
        "Returns the string representation of a Python type without metadata."

        # return forward references as the annotation string
        if isinstance(data_type, typing.ForwardRef):
            fwd: typing.ForwardRef = data_type
            return fwd.__forward_arg__
        elif isinstance(data_type, str):
            return data_type

        origin = typing.get_origin(data_type)
        if origin is not None:
            data_type_args = typing.get_args(data_type)

            if origin is dict:  # Dict[T]
                origin_name = "Dict"
            elif origin is list:  # List[T]
                origin_name = "List"
            elif origin is set:  # Set[T]
                origin_name = "Set"
            elif origin is Union:
                return self.union_to_str(data_type_args)
            elif origin is Literal:
                args = ", ".join(repr(arg) for arg in data_type_args)
                return f"Literal[{args}]"
            else:
                origin_name = origin.__name__

            args = ", ".join(self.python_type_to_str(t) for t in data_type_args)
            return f"{origin_name}[{args}]"

        return data_type.__name__

    def python_type_to_str(self, data_type: TypeLike) -> str:
        "Returns the string representation of a Python type."

        if data_type is type(None):
            return "None"

        # use compact name for alias types
        name = _auxiliary_types.get(data_type)
        if name is not None:
            return name

        metadata = getattr(data_type, "__metadata__", None)
        if metadata is not None:
            # type is Annotated[T, ...]
            metatuple: tuple[Any, ...] = metadata
            arg = typing.get_args(data_type)[0]

            # check for auxiliary types with user-defined annotations
            metaset = set(metatuple)
            for auxiliary_type, auxiliary_name in _auxiliary_types.items():
                auxiliary_arg = typing.get_args(auxiliary_type)[0]
                if arg is not auxiliary_arg:
                    continue

                auxiliary_metatuple: tuple[Any, ...] | None = getattr(auxiliary_type, "__metadata__", None)
                if auxiliary_metatuple is None:
                    continue

                if metaset.issuperset(auxiliary_metatuple):
                    # type is an auxiliary type with extra annotations
                    auxiliary_args = ", ".join(repr(m) for m in metatuple if m not in auxiliary_metatuple)
                    return f"Annotated[{auxiliary_name}, {auxiliary_args}]"

            # type is an annotated type
            args = ", ".join(repr(m) for m in metatuple)
            return f"Annotated[{self.plain_type_to_str(arg)}, {args}]"
        else:
            # type is a regular type
            return self.plain_type_to_str(data_type)


def python_type_to_str(data_type: TypeLike, use_union_operator: bool = False) -> str:
    """
    Returns the string representation of a Python type.

    :param use_union_operator: Whether to emit union types as `X | Y` as per PEP 604.
    """

    fmt = TypeFormatter(use_union_operator)
    return fmt.python_type_to_str(data_type)


def python_type_to_name(data_type: TypeLike, force: bool = False) -> str:
    """
    Returns the short name of a Python type.

    :param force: Whether to produce a name for composite types such as generics.
    """

    # use compact name for alias types
    name = _auxiliary_types.get(data_type)
    if name is not None:
        return name

    # unwrap annotated types
    metadata = getattr(data_type, "__metadata__", None)
    if metadata is not None:
        # type is Annotated[T, ...]
        arg = typing.get_args(data_type)[0]
        return python_type_to_name(arg, force=force)

    if force:
        # generic types
        if is_type_optional(data_type, strict=True):
            inner_name = python_type_to_name(unwrap_optional_type(data_type), force=True)
            return f"Optional__{inner_name}"
        elif is_generic_list(data_type):
            item_name = python_type_to_name(unwrap_generic_list(data_type), force=True)
            return f"List__{item_name}"
        elif is_generic_sequence(data_type):
            # Treat Sequence the same as List for schema generation purposes
            item_name = python_type_to_name(unwrap_generic_sequence(data_type), force=True)
            return f"List__{item_name}"
        elif is_generic_dict(data_type):
            key_type, value_type = unwrap_generic_dict(data_type)
            key_name = python_type_to_name(key_type, force=True)
            value_name = python_type_to_name(value_type, force=True)
            return f"Dict__{key_name}__{value_name}"
        elif is_type_union(data_type):
            member_types = unwrap_union_types(data_type)
            member_names = "__".join(python_type_to_name(member_type, force=True) for member_type in member_types)
            return f"Union__{member_names}"

    # named system or user-defined type
    if hasattr(data_type, "__name__") and not typing.get_args(data_type):
        return data_type.__name__

    raise TypeError(f"cannot assign a simple name to type: {data_type}")
