# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Type-safe data interchange for Python data classes.

:see: https://github.com/hunyadi/strong_typing
"""

import dataclasses
import sys
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import TypeVar, overload

if sys.version_info >= (3, 9):
    from typing import Annotated as Annotated
else:
    from typing import Annotated as Annotated

if sys.version_info >= (3, 10):
    from typing import TypeAlias as TypeAlias
else:
    from typing import TypeAlias as TypeAlias

if sys.version_info >= (3, 11):
    from typing import dataclass_transform as dataclass_transform
else:
    from typing import dataclass_transform as dataclass_transform

T = TypeVar("T")


def _compact_dataclass_repr(obj: object) -> str:
    """
    Compact data-class representation where positional arguments are used instead of keyword arguments.

    :param obj: A data-class object.
    :returns: A string that matches the pattern `Class(arg1, arg2, ...)`.
    """

    if is_dataclass(obj):
        arglist = ", ".join(repr(getattr(obj, field.name)) for field in dataclasses.fields(obj))
        return f"{obj.__class__.__name__}({arglist})"
    else:
        return obj.__class__.__name__


class CompactDataClass:
    "A data class whose repr() uses positional rather than keyword arguments."

    def __repr__(self) -> str:
        return _compact_dataclass_repr(self)


@overload
def typeannotation(cls: type[T], /) -> type[T]: ...


@overload
def typeannotation(cls: None, *, eq: bool = True, order: bool = False) -> Callable[[type[T]], type[T]]: ...


@dataclass_transform(eq_default=True, order_default=False)
def typeannotation(
    cls: type[T] | None = None, *, eq: bool = True, order: bool = False
) -> type[T] | Callable[[type[T]], type[T]]:
    """
    Returns the same class as was passed in, with dunder methods added based on the fields defined in the class.

    :param cls: The data-class type to transform into a type annotation.
    :param eq: Whether to generate functions to support equality comparison.
    :param order: Whether to generate functions to support ordering.
    :returns: A data-class type, or a wrapper for data-class types.
    """

    def wrap(cls: type[T]) -> type[T]:
        # mypy fails to equate bound-y functions (first argument interpreted as
        # the bound object) with class methods, hence the `ignore` directive.
        cls.__repr__ = _compact_dataclass_repr  # type: ignore[method-assign]
        if not dataclasses.is_dataclass(cls):
            cls = dataclasses.dataclass(  # type: ignore[call-overload]
                cls,
                init=True,
                repr=False,
                eq=eq,
                order=order,
                unsafe_hash=False,
                frozen=True,
            )
        return cls

    # see if decorator is used as @typeannotation or @typeannotation()
    if cls is None:
        # called with parentheses
        return wrap
    else:
        # called without parentheses
        return wrap(cls)


@typeannotation
class Alias:
    "Alternative name of a property, typically used in JSON serialization."

    name: str


@typeannotation
class Signed:
    "Signedness of an integer type."

    is_signed: bool


@typeannotation
class Storage:
    "Number of bytes the binary representation of an integer type takes, e.g. 4 bytes for an int32."

    bytes: int


@typeannotation
class IntegerRange:
    "Minimum and maximum value of an integer. The range is inclusive."

    minimum: int
    maximum: int


@typeannotation
class Precision:
    "Precision of a floating-point value."

    significant_digits: int
    decimal_digits: int = 0

    @property
    def integer_digits(self) -> int:
        return self.significant_digits - self.decimal_digits


@typeannotation
class TimePrecision:
    """
    Precision of a timestamp or time interval.

    :param decimal_digits: Number of fractional digits retained in the sub-seconds field for a timestamp.
    """

    decimal_digits: int = 0


@typeannotation
class Length:
    "Exact length of a string."

    value: int


@typeannotation
class MinLength:
    "Minimum length of a string."

    value: int


@typeannotation
class MaxLength:
    "Maximum length of a string."

    value: int


@typeannotation
class SpecialConversion:
    "Indicates that the annotated type is subject to custom conversion rules."


int8: TypeAlias = Annotated[int, Signed(True), Storage(1), IntegerRange(-128, 127)]
int16: TypeAlias = Annotated[int, Signed(True), Storage(2), IntegerRange(-32768, 32767)]
int32: TypeAlias = Annotated[
    int,
    Signed(True),
    Storage(4),
    IntegerRange(-2147483648, 2147483647),
]
int64: TypeAlias = Annotated[
    int,
    Signed(True),
    Storage(8),
    IntegerRange(-9223372036854775808, 9223372036854775807),
]

uint8: TypeAlias = Annotated[int, Signed(False), Storage(1), IntegerRange(0, 255)]
uint16: TypeAlias = Annotated[int, Signed(False), Storage(2), IntegerRange(0, 65535)]
uint32: TypeAlias = Annotated[
    int,
    Signed(False),
    Storage(4),
    IntegerRange(0, 4294967295),
]
uint64: TypeAlias = Annotated[
    int,
    Signed(False),
    Storage(8),
    IntegerRange(0, 18446744073709551615),
]

float32: TypeAlias = Annotated[float, Storage(4)]
float64: TypeAlias = Annotated[float, Storage(8)]

# maps globals of type Annotated[T, ...] defined in this module to their string names
_auxiliary_types: dict[object, str] = {}
module = sys.modules[__name__]
for var in dir(module):
    typ = getattr(module, var)
    if getattr(typ, "__metadata__", None) is not None:
        # type is Annotated[T, ...]
        _auxiliary_types[typ] = var


def get_auxiliary_format(data_type: object) -> str | None:
    "Returns the JSON format string corresponding to an auxiliary type."

    return _auxiliary_types.get(data_type)
