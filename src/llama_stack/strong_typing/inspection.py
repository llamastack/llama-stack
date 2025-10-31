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
import datetime
import enum
import importlib
import importlib.machinery
import importlib.util
import inspect
import re
import sys
import types
import typing
import uuid
from collections.abc import Callable, Iterable
from typing import (
    Any,
    Literal,
    NamedTuple,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    from typing import Annotated

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing import TypeGuard


from pydantic import BaseModel
from pydantic.fields import FieldInfo

S = TypeVar("S")
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


def _is_type_like(data_type: object) -> bool:
    """
    Checks if the object is a type or type-like object (e.g. generic type).

    :param data_type: The object to validate.
    :returns: True if the object is a type or type-like object.
    """

    if isinstance(data_type, type):
        # a standard type
        return True
    elif typing.get_origin(data_type) is not None:
        # a generic type such as `list`, `dict` or `set`
        return True
    elif hasattr(data_type, "__forward_arg__"):
        # an instance of `ForwardRef`
        return True
    elif data_type is Any:
        # the special form `Any`
        return True
    else:
        return False


if sys.version_info >= (3, 9):
    TypeLike = Union[type, types.GenericAlias, typing.ForwardRef, Any]

    def is_type_like(
        data_type: object,
    ) -> TypeGuard[TypeLike]:
        """
        Checks if the object is a type or type-like object (e.g. generic type).

        :param data_type: The object to validate.
        :returns: True if the object is a type or type-like object.
        """

        return _is_type_like(data_type)

else:
    TypeLike = object

    def is_type_like(
        data_type: object,
    ) -> bool:
        return _is_type_like(data_type)


def evaluate_member_type(typ: Any, cls: type) -> Any:
    """
    Evaluates a forward reference type in a dataclass member.

    :param typ: The dataclass member type to convert.
    :param cls: The dataclass in which the member is defined.
    :returns: The evaluated type.
    """

    return evaluate_type(typ, sys.modules[cls.__module__])


def evaluate_type(typ: Any, module: types.ModuleType) -> Any:
    """
    Evaluates a forward reference type.

    :param typ: The type to convert, typically a dataclass member type.
    :param module: The context for the type, i.e. the module in which the member is defined.
    :returns: The evaluated type.
    """

    if isinstance(typ, str):
        # evaluate data-class field whose type annotation is a string
        return eval(typ, module.__dict__, locals())
    if isinstance(typ, typing.ForwardRef):
        if sys.version_info >= (3, 9):
            return typ._evaluate(module.__dict__, locals(), recursive_guard=frozenset())
        else:
            return typ._evaluate(module.__dict__, locals())
    else:
        return typ


@runtime_checkable
class DataclassInstance(Protocol):
    __dataclass_fields__: typing.ClassVar[dict[str, dataclasses.Field]]


def is_dataclass_type(typ: Any) -> TypeGuard[type[DataclassInstance]]:
    "True if the argument corresponds to a data class type (but not an instance)."

    typ = unwrap_annotated_type(typ)
    return isinstance(typ, type) and dataclasses.is_dataclass(typ)


def is_dataclass_instance(obj: Any) -> TypeGuard[DataclassInstance]:
    "True if the argument corresponds to a data class instance (but not a type)."

    return not isinstance(obj, type) and dataclasses.is_dataclass(obj)


@dataclasses.dataclass
class DataclassField:
    name: str
    type: Any
    default: Any

    def __init__(self, name: str, type: Any, default: Any = dataclasses.MISSING) -> None:
        self.name = name
        self.type = type
        self.default = default


def dataclass_fields(cls: type[DataclassInstance]) -> Iterable[DataclassField]:
    "Generates the fields of a data-class resolving forward references."

    for field in dataclasses.fields(cls):
        yield DataclassField(field.name, evaluate_member_type(field.type, cls), field.default)


def dataclass_field_by_name(cls: type[DataclassInstance], name: str) -> DataclassField:
    "Looks up a field in a data-class by its field name."

    for field in dataclasses.fields(cls):
        if field.name == name:
            return DataclassField(field.name, evaluate_member_type(field.type, cls))

    raise LookupError(f"field `{name}` missing from class `{cls.__name__}`")


def is_named_tuple_instance(obj: Any) -> TypeGuard[NamedTuple]:
    "True if the argument corresponds to a named tuple instance."

    return is_named_tuple_type(type(obj))


def is_named_tuple_type(typ: Any) -> TypeGuard[type[NamedTuple]]:
    """
    True if the argument corresponds to a named tuple type.

    Calling the function `collections.namedtuple` gives a new type that is a subclass of `tuple` (and no other classes)
    with a member named `_fields` that is a tuple whose items are all strings.
    """

    if not isinstance(typ, type):
        return False

    typ = unwrap_annotated_type(typ)

    b = getattr(typ, "__bases__", None)
    if b is None:
        return False

    if len(b) != 1 or b[0] != tuple:
        return False

    f = getattr(typ, "_fields", None)
    if not isinstance(f, tuple):
        return False

    return all(isinstance(n, str) for n in f)


if sys.version_info >= (3, 11):

    def is_type_enum(typ: object) -> TypeGuard[type[enum.Enum]]:
        "True if the specified type is an enumeration type."

        typ = unwrap_annotated_type(typ)
        return isinstance(typ, enum.EnumType)

else:

    def is_type_enum(typ: object) -> TypeGuard[type[enum.Enum]]:
        "True if the specified type is an enumeration type."

        typ = unwrap_annotated_type(typ)

        # use an explicit isinstance(..., type) check to filter out special forms like generics
        return isinstance(typ, type) and issubclass(typ, enum.Enum)


def enum_value_types(enum_type: type[enum.Enum]) -> list[type]:
    """
    Returns all unique value types of the `enum.Enum` type in definition order.
    """

    # filter unique enumeration value types by keeping definition order
    return list(dict.fromkeys(type(e.value) for e in enum_type))


def extend_enum(
    source: type[enum.Enum],
) -> Callable[[type[enum.Enum]], type[enum.Enum]]:
    """
    Creates a new enumeration type extending the set of values in an existing type.

    :param source: The existing enumeration type to be extended with new values.
    :returns: A new enumeration type with the extended set of values.
    """

    def wrap(extend: type[enum.Enum]) -> type[enum.Enum]:
        # create new enumeration type combining the values from both types
        values: dict[str, Any] = {}
        values.update((e.name, e.value) for e in source)
        values.update((e.name, e.value) for e in extend)
        # mypy fails to determine that __name__ is always a string; hence the `ignore` directive.
        enum_class: type[enum.Enum] = enum.Enum(extend.__name__, values)  # type: ignore[misc]

        # assign the newly created type to the same module where the extending class is defined
        enum_class.__module__ = extend.__module__
        enum_class.__doc__ = extend.__doc__
        setattr(sys.modules[extend.__module__], extend.__name__, enum_class)

        return enum.unique(enum_class)

    return wrap


if sys.version_info >= (3, 10):

    def _is_union_like(typ: object) -> bool:
        "True if type is a union such as `Union[T1, T2, ...]` or a union type `T1 | T2`."

        return typing.get_origin(typ) is Union or isinstance(typ, types.UnionType)

else:

    def _is_union_like(typ: object) -> bool:
        "True if type is a union such as `Union[T1, T2, ...]` or a union type `T1 | T2`."

        return typing.get_origin(typ) is Union


def is_type_optional(typ: object, strict: bool = False) -> TypeGuard[type[Any | None]]:
    """
    True if the type annotation corresponds to an optional type (e.g. `Optional[T]` or `Union[T1,T2,None]`).

    `Optional[T]` is represented as `Union[T, None]` is classic style, and is equivalent to `T | None` in new style.

    :param strict: True if only `Optional[T]` qualifies as an optional type but `Union[T1, T2, None]` does not.
    """

    typ = unwrap_annotated_type(typ)

    if _is_union_like(typ):
        args = typing.get_args(typ)
        if strict and len(args) != 2:
            return False

        return type(None) in args

    return False


def unwrap_optional_type(typ: type[T | None]) -> type[T]:
    """
    Extracts the inner type of an optional type.

    :param typ: The optional type `Optional[T]`.
    :returns: The inner type `T`.
    """

    return rewrap_annotated_type(_unwrap_optional_type, typ)


def _unwrap_optional_type(typ: type[T | None]) -> type[T]:
    "Extracts the type qualified as optional (e.g. returns `T` for `Optional[T]`)."

    # Optional[T] is represented internally as Union[T, None]
    if not _is_union_like(typ):
        raise TypeError("optional type must have un-subscripted type of Union")

    # will automatically unwrap Union[T] into T
    return Union[tuple(filter(lambda item: item is not type(None), typing.get_args(typ)))]  # type: ignore[return-value]


def is_type_union(typ: object) -> bool:
    "True if the type annotation corresponds to a union type (e.g. `Union[T1,T2,T3]`)."

    typ = unwrap_annotated_type(typ)
    if _is_union_like(typ):
        args = typing.get_args(typ)
        return len(args) > 2 or type(None) not in args

    return False


def unwrap_union_types(typ: object) -> tuple[object, ...]:
    """
    Extracts the inner types of a union type.

    :param typ: The union type `Union[T1, T2, ...]`.
    :returns: The inner types `T1`, `T2`, etc.
    """

    typ = unwrap_annotated_type(typ)
    return _unwrap_union_types(typ)


def _unwrap_union_types(typ: object) -> tuple[object, ...]:
    "Extracts the types in a union (e.g. returns a tuple of types `T1` and `T2` for `Union[T1, T2]`)."

    if not _is_union_like(typ):
        raise TypeError("union type must have un-subscripted type of Union")

    return typing.get_args(typ)


def is_type_literal(typ: object) -> bool:
    "True if the specified type is a literal of one or more constant values, e.g. `Literal['string']` or `Literal[42]`."

    typ = unwrap_annotated_type(typ)
    return typing.get_origin(typ) is Literal


def unwrap_literal_value(typ: object) -> Any:
    """
    Extracts the single constant value captured by a literal type.

    :param typ: The literal type `Literal[value]`.
    :returns: The values captured by the literal type.
    """

    args = unwrap_literal_values(typ)
    if len(args) != 1:
        raise TypeError("too many values in literal type")

    return args[0]


def unwrap_literal_values(typ: object) -> tuple[Any, ...]:
    """
    Extracts the constant values captured by a literal type.

    :param typ: The literal type `Literal[value, ...]`.
    :returns: A tuple of values captured by the literal type.
    """

    typ = unwrap_annotated_type(typ)
    return typing.get_args(typ)


def unwrap_literal_types(typ: object) -> tuple[type, ...]:
    """
    Extracts the types of the constant values captured by a literal type.

    :param typ: The literal type `Literal[value, ...]`.
    :returns: A tuple of item types `T` such that `type(value) == T`.
    """

    return tuple(type(t) for t in unwrap_literal_values(typ))


def is_generic_list(typ: object) -> TypeGuard[type[list]]:
    "True if the specified type is a generic list, i.e. `List[T]`."

    typ = unwrap_annotated_type(typ)
    return typing.get_origin(typ) is list


def unwrap_generic_list(typ: type[list[T]]) -> type[T]:
    """
    Extracts the item type of a list type.

    :param typ: The list type `List[T]`.
    :returns: The item type `T`.
    """

    return rewrap_annotated_type(_unwrap_generic_list, typ)


def _unwrap_generic_list(typ: type[list[T]]) -> type[T]:
    "Extracts the item type of a list type (e.g. returns `T` for `List[T]`)."

    (list_type,) = typing.get_args(typ)  # unpack single tuple element
    return list_type  # type: ignore[no-any-return]


def is_generic_sequence(typ: object) -> bool:
    "True if the specified type is a generic Sequence, i.e. `Sequence[T]`."
    import collections.abc

    typ = unwrap_annotated_type(typ)
    return typing.get_origin(typ) is collections.abc.Sequence


def unwrap_generic_sequence(typ: object) -> type:
    """
    Extracts the item type of a Sequence type.

    :param typ: The Sequence type `Sequence[T]`.
    :returns: The item type `T`.
    """

    return rewrap_annotated_type(_unwrap_generic_sequence, typ)  # type: ignore[arg-type]


def _unwrap_generic_sequence(typ: object) -> type:
    "Extracts the item type of a Sequence type (e.g. returns `T` for `Sequence[T]`)."

    (sequence_type,) = typing.get_args(typ)  # unpack single tuple element
    return sequence_type  # type: ignore[no-any-return]


def is_generic_set(typ: object) -> TypeGuard[type[set]]:
    "True if the specified type is a generic set, i.e. `Set[T]`."

    typ = unwrap_annotated_type(typ)
    return typing.get_origin(typ) is set


def unwrap_generic_set(typ: type[set[T]]) -> type[T]:
    """
    Extracts the item type of a set type.

    :param typ: The set type `Set[T]`.
    :returns: The item type `T`.
    """

    return rewrap_annotated_type(_unwrap_generic_set, typ)


def _unwrap_generic_set(typ: type[set[T]]) -> type[T]:
    "Extracts the item type of a set type (e.g. returns `T` for `Set[T]`)."

    (set_type,) = typing.get_args(typ)  # unpack single tuple element
    return set_type  # type: ignore[no-any-return]


def is_generic_dict(typ: object) -> TypeGuard[type[dict]]:
    "True if the specified type is a generic dictionary, i.e. `Dict[KeyType, ValueType]`."

    typ = unwrap_annotated_type(typ)
    return typing.get_origin(typ) is dict


def unwrap_generic_dict(typ: type[dict[K, V]]) -> tuple[type[K], type[V]]:
    """
    Extracts the key and value types of a dictionary type as a tuple.

    :param typ: The dictionary type `Dict[K, V]`.
    :returns: The key and value types `K` and `V`.
    """

    return _unwrap_generic_dict(unwrap_annotated_type(typ))


def _unwrap_generic_dict(typ: type[dict[K, V]]) -> tuple[type[K], type[V]]:
    "Extracts the key and value types of a dict type (e.g. returns (`K`, `V`) for `Dict[K, V]`)."

    key_type, value_type = typing.get_args(typ)
    return key_type, value_type


def is_type_annotated(typ: TypeLike) -> bool:
    "True if the type annotation corresponds to an annotated type (i.e. `Annotated[T, ...]`)."

    return getattr(typ, "__metadata__", None) is not None


def get_annotation(data_type: TypeLike, annotation_type: type[T]) -> T | None:
    """
    Returns the first annotation on a data type that matches the expected annotation type.

    :param data_type: The annotated type from which to extract the annotation.
    :param annotation_type: The annotation class to look for.
    :returns: The annotation class instance found (if any).
    """

    metadata = getattr(data_type, "__metadata__", None)
    if metadata is not None:
        for annotation in metadata:
            if isinstance(annotation, annotation_type):
                return annotation

    return None


def unwrap_annotated_type(typ: T) -> T:
    "Extracts the wrapped type from an annotated type (e.g. returns `T` for `Annotated[T, ...]`)."

    if is_type_annotated(typ):
        # type is Annotated[T, ...]
        return typing.get_args(typ)[0]  # type: ignore[no-any-return]
    else:
        # type is a regular type
        return typ


def rewrap_annotated_type(transform: Callable[[type[S]], type[T]], typ: type[S]) -> type[T]:
    """
    Un-boxes, transforms and re-boxes an optionally annotated type.

    :param transform: A function that maps an un-annotated type to another type.
    :param typ: A type to un-box (if necessary), transform, and re-box (if necessary).
    """

    metadata = getattr(typ, "__metadata__", None)
    if metadata is not None:
        # type is Annotated[T, ...]
        inner_type = typing.get_args(typ)[0]
    else:
        # type is a regular type
        inner_type = typ

    transformed_type = transform(inner_type)

    if metadata is not None:
        return Annotated[(transformed_type, *metadata)]  # type: ignore[return-value]
    else:
        return transformed_type


def get_module_classes(module: types.ModuleType) -> list[type]:
    "Returns all classes declared directly in a module."

    def is_class_member(member: object) -> TypeGuard[type]:
        return inspect.isclass(member) and member.__module__ == module.__name__

    return [class_type for _, class_type in inspect.getmembers(module, is_class_member)]


if sys.version_info >= (3, 9):

    def get_resolved_hints(typ: type) -> dict[str, type]:
        return typing.get_type_hints(typ, include_extras=True)

else:

    def get_resolved_hints(typ: type) -> dict[str, type]:
        return typing.get_type_hints(typ)


def get_class_properties(typ: type) -> Iterable[tuple[str, type | str]]:
    "Returns all properties of a class."

    if is_dataclass_type(typ):
        return ((field.name, field.type) for field in dataclasses.fields(typ))
    elif hasattr(typ, "model_fields"):
        # Pydantic BaseModel - use model_fields to exclude ClassVar and other non-field attributes
        # Reconstruct Annotated type if discriminator exists to preserve metadata
        from typing import Annotated, Any

        from pydantic.fields import FieldInfo

        def get_field_type(name: str, field: Any) -> type | str:
            # If field has discriminator, wrap in Annotated to preserve it for schema generation
            if field.discriminator:
                field_info = FieldInfo(annotation=None, discriminator=field.discriminator)
                # Annotated returns _AnnotatedAlias which isn't a type but is valid here
                return Annotated[field.annotation, field_info]  # type: ignore[return-value]
            # field.annotation can be Union types, Annotated, etc. which aren't type but are valid
            return field.annotation  # type: ignore[return-value,no-any-return]

        return ((name, get_field_type(name, field)) for name, field in typ.model_fields.items())
    else:
        resolved_hints = get_resolved_hints(typ)
        return resolved_hints.items()


def get_class_property(typ: type, name: str) -> type | str | None:
    "Looks up the annotated type of a property in a class by its property name."

    for property_name, property_type in get_class_properties(typ):
        if name == property_name:
            return property_type
    return None


@dataclasses.dataclass
class _ROOT:
    pass


def get_referenced_types(typ: TypeLike, module: types.ModuleType | None = None) -> set[type]:
    """
    Extracts types directly or indirectly referenced by this type.

    For example, extract `T` from `List[T]`, `Optional[T]` or `Annotated[T, ...]`, `K` and `V` from `Dict[K,V]`,
    `A` and `B` from `Union[A,B]`.

    :param typ: A type or special form.
    :param module: The context in which types are evaluated.
    :returns: Types referenced by the given type or special form.
    """

    collector = TypeCollector()
    collector.run(typ, _ROOT, module)
    return collector.references


class TypeCollector:
    """
    Collects types directly or indirectly referenced by a type.

    :param graph: The type dependency graph, linking types to types they depend on.
    """

    graph: dict[type, set[type]]

    @property
    def references(self) -> set[type]:
        "Types collected by the type collector."

        dependencies = set()
        for edges in self.graph.values():
            dependencies.update(edges)
        return dependencies

    def __init__(self) -> None:
        self.graph = {_ROOT: set()}

    def traverse(self, typ: type) -> None:
        "Finds all dependent types of a type."

        self.run(typ, _ROOT, sys.modules[typ.__module__])

    def traverse_all(self, types: Iterable[type]) -> None:
        "Finds all dependent types of a list of types."

        for typ in types:
            self.traverse(typ)

    def run(
        self,
        typ: TypeLike,
        cls: type[DataclassInstance],
        module: types.ModuleType | None,
    ) -> None:
        """
        Extracts types indirectly referenced by this type.

        For example, extract `T` from `List[T]`, `Optional[T]` or `Annotated[T, ...]`, `K` and `V` from `Dict[K,V]`,
        `A` and `B` from `Union[A,B]`.

        :param typ: A type or special form.
        :param cls: A dataclass type being expanded for dependent types.
        :param module: The context in which types are evaluated.
        :returns: Types referenced by the given type or special form.
        """

        if typ is type(None) or typ is Any:
            return

        if isinstance(typ, type):
            self.graph[cls].add(typ)

            if typ in self.graph:
                return

            self.graph[typ] = set()

        metadata = getattr(typ, "__metadata__", None)
        if metadata is not None:
            # type is Annotated[T, ...]
            arg = typing.get_args(typ)[0]
            return self.run(arg, cls, module)

        # type is a forward reference
        if isinstance(typ, str) or isinstance(typ, typing.ForwardRef):
            if module is None:
                raise ValueError("missing context for evaluating types")

            evaluated_type = evaluate_type(typ, module)
            return self.run(evaluated_type, cls, module)

        # type is a special form
        origin = typing.get_origin(typ)
        if origin in [list, dict, frozenset, set, tuple, Union]:
            for arg in typing.get_args(typ):
                self.run(arg, cls, module)
            return
        elif origin is Literal:
            return

        # type is optional or a union type
        if is_type_optional(typ):
            return self.run(unwrap_optional_type(typ), cls, module)
        if is_type_union(typ):
            for union_type in unwrap_union_types(typ):
                self.run(union_type, cls, module)
            return

        # type is a regular type
        elif is_dataclass_type(typ) or is_type_enum(typ) or isinstance(typ, type):
            context = sys.modules[typ.__module__]
            if is_dataclass_type(typ):
                for field in dataclass_fields(typ):
                    self.run(field.type, typ, context)
            else:
                for field_name, field_type in get_resolved_hints(typ).items():
                    self.run(field_type, typ, context)
            return

        raise TypeError(f"expected: type-like; got: {typ}")


if sys.version_info >= (3, 10):

    def get_signature(fn: Callable[..., Any]) -> inspect.Signature:
        "Extracts the signature of a function."

        return inspect.signature(fn, eval_str=True)

else:

    def get_signature(fn: Callable[..., Any]) -> inspect.Signature:
        "Extracts the signature of a function."

        return inspect.signature(fn)


def is_reserved_property(name: str) -> bool:
    "True if the name stands for an internal property."

    # filter built-in and special properties
    if re.match(r"^__.+__$", name):
        return True

    # filter built-in special names
    if name in ["_abc_impl"]:
        return True

    return False


def create_module(name: str) -> types.ModuleType:
    """
    Creates a new module dynamically at run-time.

    :param name: Fully qualified name of the new module (with dot notation).
    """

    if name in sys.modules:
        raise KeyError(f"{name!r} already in sys.modules")

    spec = importlib.machinery.ModuleSpec(name, None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if spec.loader is not None:
        spec.loader.exec_module(module)
    return module


if sys.version_info >= (3, 10):

    def create_data_type(class_name: str, fields: list[tuple[str, type]]) -> type:
        """
        Creates a new data-class type dynamically.

        :param class_name: The name of new data-class type.
        :param fields: A list of fields (and their type) that the new data-class type is expected to have.
        :returns: The newly created data-class type.
        """

        # has the `slots` parameter
        return dataclasses.make_dataclass(class_name, fields, slots=True)

else:

    def create_data_type(class_name: str, fields: list[tuple[str, type]]) -> type:
        """
        Creates a new data-class type dynamically.

        :param class_name: The name of new data-class type.
        :param fields: A list of fields (and their type) that the new data-class type is expected to have.
        :returns: The newly created data-class type.
        """

        cls = dataclasses.make_dataclass(class_name, fields)

        cls_dict = dict(cls.__dict__)
        field_names = tuple(field.name for field in dataclasses.fields(cls))

        cls_dict["__slots__"] = field_names

        for field_name in field_names:
            cls_dict.pop(field_name, None)
        cls_dict.pop("__dict__", None)

        qualname = getattr(cls, "__qualname__", None)
        cls = type(cls)(cls.__name__, (), cls_dict)
        if qualname is not None:
            cls.__qualname__ = qualname

        return cls


def create_object(typ: type[T]) -> T:
    "Creates an instance of a type."

    if issubclass(typ, Exception):
        # exception types need special treatment
        e = typ.__new__(typ)
        return typing.cast(T, e)
    else:
        return object.__new__(typ)


if sys.version_info >= (3, 9):
    TypeOrGeneric = Union[type, types.GenericAlias]

else:
    TypeOrGeneric = object


def is_generic_instance(obj: Any, typ: TypeLike) -> bool:
    """
    Returns whether an object is an instance of a generic class, a standard class or of a subclass thereof.

    This function checks the following items recursively:
    * items of a list
    * keys and values of a dictionary
    * members of a set
    * items of a tuple
    * members of a union type

    :param obj: The (possibly generic container) object to check recursively.
    :param typ: The expected type of the object.
    """

    if isinstance(typ, typing.ForwardRef):
        fwd: typing.ForwardRef = typ
        identifier = fwd.__forward_arg__
        typ = eval(identifier)
        if isinstance(typ, type):
            return isinstance(obj, typ)
        else:
            return False

    # generic types (e.g. list, dict, set, etc.)
    origin_type = typing.get_origin(typ)
    if origin_type is list:
        if not isinstance(obj, list):
            return False
        (list_item_type,) = typing.get_args(typ)  # unpack single tuple element
        list_obj: list = obj
        return all(is_generic_instance(item, list_item_type) for item in list_obj)
    elif origin_type is dict:
        if not isinstance(obj, dict):
            return False
        key_type, value_type = typing.get_args(typ)
        dict_obj: dict = obj
        return all(
            is_generic_instance(key, key_type) and is_generic_instance(value, value_type)
            for key, value in dict_obj.items()
        )
    elif origin_type is set:
        if not isinstance(obj, set):
            return False
        (set_member_type,) = typing.get_args(typ)  # unpack single tuple element
        set_obj: set = obj
        return all(is_generic_instance(item, set_member_type) for item in set_obj)
    elif origin_type is tuple:
        if not isinstance(obj, tuple):
            return False
        return all(
            is_generic_instance(item, tuple_item_type)
            for tuple_item_type, item in zip(
                (tuple_item_type for tuple_item_type in typing.get_args(typ)),
                (item for item in obj),
                strict=False,
            )
        )
    elif origin_type is Union:
        return any(is_generic_instance(obj, member_type) for member_type in typing.get_args(typ))
    elif isinstance(typ, type):
        return isinstance(obj, typ)
    else:
        raise TypeError(f"expected `type` but got: {typ}")


class RecursiveChecker:
    _pred: Callable[[type, Any], bool] | None

    def __init__(self, pred: Callable[[type, Any], bool]) -> None:
        """
        Creates a checker to verify if a predicate applies to all nested member properties of an object recursively.

        :param pred: The predicate to test on member properties. Takes a property type and a property value.
        """

        self._pred = pred

    def pred(self, typ: type, obj: Any) -> bool:
        "Acts as a workaround for the type checker mypy."

        assert self._pred is not None
        return self._pred(typ, obj)

    def check(self, typ: TypeLike, obj: Any) -> bool:
        """
        Checks if a predicate applies to all nested member properties of an object recursively.

        :param typ: The type to recurse into.
        :param obj: The object to inspect recursively. Must be an instance of the given type.
        :returns: True if all member properties pass the filter predicate.
        """

        # check for well-known types
        if (
            typ is type(None)
            or typ is bool
            or typ is int
            or typ is float
            or typ is str
            or typ is bytes
            or typ is datetime.datetime
            or typ is datetime.date
            or typ is datetime.time
            or typ is uuid.UUID
        ):
            return self.pred(typing.cast(type, typ), obj)

        # generic types (e.g. list, dict, set, etc.)
        origin_type = typing.get_origin(typ)
        if origin_type is list:
            if not isinstance(obj, list):
                raise TypeError(f"expected `list` but got: {obj}")
            (list_item_type,) = typing.get_args(typ)  # unpack single tuple element
            list_obj: list = obj
            return all(self.check(list_item_type, item) for item in list_obj)
        elif origin_type is dict:
            if not isinstance(obj, dict):
                raise TypeError(f"expected `dict` but got: {obj}")
            key_type, value_type = typing.get_args(typ)
            dict_obj: dict = obj
            return all(self.check(value_type, item) for item in dict_obj.values())
        elif origin_type is set:
            if not isinstance(obj, set):
                raise TypeError(f"expected `set` but got: {obj}")
            (set_member_type,) = typing.get_args(typ)  # unpack single tuple element
            set_obj: set = obj
            return all(self.check(set_member_type, item) for item in set_obj)
        elif origin_type is tuple:
            if not isinstance(obj, tuple):
                raise TypeError(f"expected `tuple` but got: {obj}")
            return all(
                self.check(tuple_item_type, item)
                for tuple_item_type, item in zip(
                    (tuple_item_type for tuple_item_type in typing.get_args(typ)),
                    (item for item in obj),
                    strict=False,
                )
            )
        elif origin_type is Union:
            return self.pred(typ, obj)  # type: ignore[arg-type]

        if not inspect.isclass(typ):
            raise TypeError(f"expected `type` but got: {typ}")

        # enumeration type
        if issubclass(typ, enum.Enum):
            if not isinstance(obj, enum.Enum):
                raise TypeError(f"expected `{typ}` but got: {obj}")
            return self.pred(typ, obj)

        # class types with properties
        if is_named_tuple_type(typ):
            if not isinstance(obj, tuple):
                raise TypeError(f"expected `NamedTuple` but got: {obj}")
            return all(
                self.check(field_type, getattr(obj, field_name))
                for field_name, field_type in typing.get_type_hints(typ).items()
            )
        elif is_dataclass_type(typ):
            if not isinstance(obj, typ):
                raise TypeError(f"expected `{typ}` but got: {obj}")
            resolved_hints = get_resolved_hints(typ)
            return all(
                self.check(resolved_hints[field.name], getattr(obj, field.name)) for field in dataclasses.fields(typ)
            )
        else:
            if not isinstance(obj, typ):
                raise TypeError(f"expected `{typ}` but got: {obj}")
            return all(
                self.check(property_type, getattr(obj, property_name))
                for property_name, property_type in get_class_properties(typ)
            )


def check_recursive(
    obj: object,
    /,
    *,
    pred: Callable[[type, Any], bool] | None = None,
    type_pred: Callable[[type], bool] | None = None,
    value_pred: Callable[[Any], bool] | None = None,
) -> bool:
    """
    Checks if a predicate applies to all nested member properties of an object recursively.

    :param obj: The object to inspect recursively.
    :param pred: The predicate to test on member properties. Takes a property type and a property value.
    :param type_pred: Constrains the check to properties of an expected type. Properties of other types pass automatically.
    :param value_pred: Verifies a condition on member property values (of an expected type).
    :returns: True if all member properties pass the filter predicate(s).
    """

    if type_pred is not None and value_pred is not None:
        if pred is not None:
            raise TypeError("filter predicate not permitted when type and value predicates are present")

        type_p: Callable[[type[T]], bool] = type_pred
        value_p: Callable[[T], bool] = value_pred
        pred = lambda typ, obj: not type_p(typ) or value_p(obj)  # noqa: E731

    elif value_pred is not None:
        if pred is not None:
            raise TypeError("filter predicate not permitted when value predicate is present")

        value_only_p: Callable[[T], bool] = value_pred
        pred = lambda typ, obj: value_only_p(obj)  # noqa: E731

    elif type_pred is not None:
        raise TypeError("value predicate required when type predicate is present")

    elif pred is None:
        pred = lambda typ, obj: True  # noqa: E731

    return RecursiveChecker(pred).check(type(obj), obj)


def is_unwrapped_body_param(param_type: Any) -> bool:
    """
    Check if a parameter type represents an unwrapped body parameter.
    An unwrapped body parameter is an Annotated type with Body(embed=False)

    This is used to determine whether request parameters should be flattened
    in OpenAPI specs and client libraries (matching FastAPI's embed=False behavior).

    Args:
        param_type: The parameter type annotation to check

    Returns:
        True if the parameter should be treated as an unwrapped body parameter
    """
    # Check if it's Annotated with Body(embed=False)
    if typing.get_origin(param_type) is Annotated:
        args = typing.get_args(param_type)
        base_type = args[0]
        metadata = args[1:]

        # Look for Body annotation with embed=False
        # Body() returns a FieldInfo object, so we check for that type and the embed attribute
        for item in metadata:
            if isinstance(item, FieldInfo) and hasattr(item, "embed") and not item.embed:
                return inspect.isclass(base_type) and issubclass(base_type, BaseModel)

    return False
