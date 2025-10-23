# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Tuple, Type, TypeVar

T = TypeVar("T")


class SlotsMeta(type):
    def __new__(cls: Type[T], name: str, bases: Tuple[type, ...], ns: Dict[str, Any]) -> T:
        # caller may have already provided slots, in which case just retain them and keep going
        slots: Tuple[str, ...] = ns.get("__slots__", ())

        # add fields with type annotations to slots
        annotations: Dict[str, Any] = ns.get("__annotations__", {})
        members = tuple(member for member in annotations.keys() if member not in slots)

        # assign slots
        ns["__slots__"] = slots + tuple(members)
        return super().__new__(cls, name, bases, ns)  # type: ignore


class Slots(metaclass=SlotsMeta):
    pass
