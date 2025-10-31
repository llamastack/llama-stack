# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Type-safe data interchange for Python data classes.

:see: https://github.com/hunyadi/strong_typing
"""

import builtins
import collections.abc
import dataclasses
import inspect
import re
import sys
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass
from io import StringIO
from typing import Any, Protocol, TypeVar

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing import TypeGuard

from .inspection import (
    DataclassInstance,
    get_class_properties,
    get_signature,
    is_dataclass_type,
    is_type_enum,
)

T = TypeVar("T")


@dataclass
class DocstringParam:
    """
    A parameter declaration in a parameter block.

    :param name: The name of the parameter.
    :param description: The description text for the parameter.
    """

    name: str
    description: str
    param_type: type | str = inspect.Signature.empty

    def __str__(self) -> str:
        return f":param {self.name}: {self.description}"


@dataclass
class DocstringReturns:
    """
    A `returns` declaration extracted from a docstring.

    :param description: The description text for the return value.
    """

    description: str
    return_type: type = inspect.Signature.empty

    def __str__(self) -> str:
        return f":returns: {self.description}"


@dataclass
class DocstringRaises:
    """
    A `raises` declaration extracted from a docstring.

    :param typename: The type name of the exception raised.
    :param description: The description associated with the exception raised.
    """

    typename: str
    description: str
    raise_type: type = inspect.Signature.empty

    def __str__(self) -> str:
        return f":raises {self.typename}: {self.description}"


@dataclass
class Docstring:
    """
    Represents the documentation string (a.k.a. docstring) for a type such as a (data) class or function.

    A docstring is broken down into the following components:
    * A short description, which is the first block of text in the documentation string, and ends with a double
      newline or a parameter block.
    * A long description, which is the optional block of text following the short description, and ends with
      a parameter block.
    * A parameter block of named parameter and description string pairs in ReST-style.
    * A `returns` declaration, which adds explanation to the return value.
    * A `raises` declaration, which adds explanation to the exception type raised by the function on error.

    When the docstring is attached to a data class, it is understood as the documentation string of the class
    `__init__` method.

    :param short_description: The short description text parsed from a docstring.
    :param long_description: The long description text parsed from a docstring.
    :param params: The parameter block extracted from a docstring.
    :param returns: The returns declaration extracted from a docstring.
    """

    short_description: str | None = None
    long_description: str | None = None
    params: dict[str, DocstringParam] = dataclasses.field(default_factory=dict)
    returns: DocstringReturns | None = None
    raises: dict[str, DocstringRaises] = dataclasses.field(default_factory=dict)

    @property
    def full_description(self) -> str | None:
        if self.short_description and self.long_description:
            return f"{self.short_description}\n\n{self.long_description}"
        elif self.short_description:
            return self.short_description
        else:
            return None

    def __str__(self) -> str:
        output = StringIO()

        has_description = self.short_description or self.long_description
        has_blocks = self.params or self.returns or self.raises

        if has_description:
            if self.short_description and self.long_description:
                output.write(self.short_description)
                output.write("\n\n")
                output.write(self.long_description)
            elif self.short_description:
                output.write(self.short_description)

        if has_blocks:
            if has_description:
                output.write("\n")

            for param in self.params.values():
                output.write("\n")
                output.write(str(param))
            if self.returns:
                output.write("\n")
                output.write(str(self.returns))
            for raises in self.raises.values():
                output.write("\n")
                output.write(str(raises))

        s = output.getvalue()
        output.close()
        return s


def is_exception(member: object) -> TypeGuard[type[BaseException]]:
    return isinstance(member, type) and issubclass(member, BaseException)


def get_exceptions(module: types.ModuleType) -> dict[str, type[BaseException]]:
    "Returns all exception classes declared in a module."

    return {name: class_type for name, class_type in inspect.getmembers(module, is_exception)}


class SupportsDoc(Protocol):
    __doc__: str | None


def _maybe_unwrap_async_iterator(t):
    origin_type = typing.get_origin(t)
    if origin_type is collections.abc.AsyncIterator:
        return typing.get_args(t)[0]
    return t


def parse_type(typ: SupportsDoc) -> Docstring:
    """
    Parse the docstring of a type into its components.

    :param typ: The type whose documentation string to parse.
    :returns: Components of the documentation string.
    """
    # Use docstring from the iterator origin type for streaming apis
    typ = _maybe_unwrap_async_iterator(typ)

    doc = get_docstring(typ)
    if doc is None:
        return Docstring()

    docstring = parse_text(doc)
    check_docstring(typ, docstring)

    # assign parameter and return types
    if is_dataclass_type(typ):
        properties = dict(get_class_properties(typing.cast(type, typ)))

        for name, param in docstring.params.items():
            param.param_type = properties[name]

    elif inspect.isfunction(typ):
        signature = get_signature(typ)
        for name, param in docstring.params.items():
            param.param_type = signature.parameters[name].annotation
        if docstring.returns:
            docstring.returns.return_type = signature.return_annotation

    # assign exception types
    defining_module = inspect.getmodule(typ)
    if defining_module:
        context: dict[str, type] = {}
        context.update(get_exceptions(builtins))
        context.update(get_exceptions(defining_module))
        for exc_name, exc in docstring.raises.items():
            raise_type = context.get(exc_name)
            if raise_type is None:
                type_name = getattr(typ, "__qualname__", None) or getattr(typ, "__name__", None) or None
                raise TypeError(
                    f"doc-string exception type `{exc_name}` is not an exception defined in the context of `{type_name}`"
                )

            exc.raise_type = raise_type

    return docstring


def parse_text(text: str) -> Docstring:
    """
    Parse a ReST-style docstring into its components.

    :param text: The documentation string to parse, typically acquired as `type.__doc__`.
    :returns: Components of the documentation string.
    """

    if not text:
        return Docstring()

    # find block that starts object metadata block (e.g. `:param p:` or `:returns:`)
    text = inspect.cleandoc(text)
    match = re.search("^:", text, flags=re.MULTILINE)
    if match:
        desc_chunk = text[: match.start()]
        meta_chunk = text[match.start() :]  # noqa: E203
    else:
        desc_chunk = text
        meta_chunk = ""

    # split description text into short and long description
    parts = desc_chunk.split("\n\n", 1)

    # ensure short description has no newlines
    short_description = parts[0].strip().replace("\n", " ") or None

    # ensure long description preserves its structure (e.g. preformatted text)
    if len(parts) > 1:
        long_description = parts[1].strip() or None
    else:
        long_description = None

    params: dict[str, DocstringParam] = {}
    raises: dict[str, DocstringRaises] = {}
    returns = None
    for match in re.finditer(r"(^:.*?)(?=^:|\Z)", meta_chunk, flags=re.DOTALL | re.MULTILINE):
        chunk = match.group(0)
        if not chunk:
            continue

        args_chunk, desc_chunk = chunk.lstrip(":").split(":", 1)
        args = args_chunk.split()
        desc = re.sub(r"\s+", " ", desc_chunk.strip())

        if len(args) > 0:
            kw = args[0]
            if len(args) == 2:
                if kw == "param":
                    params[args[1]] = DocstringParam(
                        name=args[1],
                        description=desc,
                    )
                elif kw == "raise" or kw == "raises":
                    raises[args[1]] = DocstringRaises(
                        typename=args[1],
                        description=desc,
                    )

            elif len(args) == 1:
                if kw == "return" or kw == "returns":
                    returns = DocstringReturns(description=desc)

    return Docstring(
        long_description=long_description,
        short_description=short_description,
        params=params,
        returns=returns,
        raises=raises,
    )


def has_default_docstring(typ: SupportsDoc) -> bool:
    "Check if class has the auto-generated string assigned by @dataclass."

    if not isinstance(typ, type):
        return False

    if is_dataclass_type(typ):
        return typ.__doc__ is not None and re.match(f"^{re.escape(typ.__name__)}[(].*[)]$", typ.__doc__) is not None

    if is_type_enum(typ):
        return typ.__doc__ is not None and typ.__doc__ == "An enumeration."

    return False


def has_docstring(typ: SupportsDoc) -> bool:
    "Check if class has a documentation string other than the auto-generated string assigned by @dataclass."

    if has_default_docstring(typ):
        return False

    return bool(typ.__doc__)


def get_docstring(typ: SupportsDoc) -> str | None:
    if typ.__doc__ is None:
        return None

    if has_default_docstring(typ):
        return None

    return typ.__doc__


def check_docstring(typ: SupportsDoc, docstring: Docstring, strict: bool = False) -> None:
    """
    Verifies the doc-string of a type.

    :raises TypeError: Raised on a mismatch between doc-string parameters, and function or type signature.
    """

    if is_dataclass_type(typ):
        check_dataclass_docstring(typ, docstring, strict)
    elif inspect.isfunction(typ):
        check_function_docstring(typ, docstring, strict)


def check_dataclass_docstring(typ: type[DataclassInstance], docstring: Docstring, strict: bool = False) -> None:
    """
    Verifies the doc-string of a data-class type.

    :param strict: Whether to check if all data-class members have doc-strings.
    :raises TypeError: Raised on a mismatch between doc-string parameters and data-class members.
    """

    if not is_dataclass_type(typ):
        raise TypeError("not a data-class type")

    properties = dict(get_class_properties(typ))
    class_name = typ.__name__

    for name in docstring.params:
        if name not in properties:
            raise TypeError(f"doc-string parameter `{name}` is not a member of the data-class `{class_name}`")

    if not strict:
        return

    for name in properties:
        if name not in docstring.params:
            raise TypeError(f"member `{name}` in data-class `{class_name}` is missing its doc-string")


def check_function_docstring(fn: Callable[..., Any], docstring: Docstring, strict: bool = False) -> None:
    """
    Verifies the doc-string of a function or member function.

    :param strict: Whether to check if all function parameters and the return type have doc-strings.
    :raises TypeError: Raised on a mismatch between doc-string parameters and function signature.
    """

    signature = get_signature(fn)
    func_name = fn.__qualname__

    for name in docstring.params:
        if name not in signature.parameters:
            raise TypeError(f"doc-string parameter `{name}` is absent from signature of function `{func_name}`")

    if docstring.returns is not None and signature.return_annotation is inspect.Signature.empty:
        raise TypeError(f"doc-string has returns description in function `{func_name}` with no return type annotation")

    if not strict:
        return

    for name, param in signature.parameters.items():
        # ignore `self` in member function signatures
        if name == "self" and (
            param.kind is inspect.Parameter.POSITIONAL_ONLY or param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ):
            continue

        if name not in docstring.params:
            raise TypeError(f"function parameter `{name}` in `{func_name}` is missing its doc-string")

    if signature.return_annotation is not inspect.Signature.empty and docstring.returns is None:
        raise TypeError(f"function `{func_name}` has no returns description in its doc-string")
