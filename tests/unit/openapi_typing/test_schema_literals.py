# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Literal

import pytest

from llama_stack.strong_typing.schema import JsonSchemaGenerator


def test_single_literal_generates_const_schema():
    gen = JsonSchemaGenerator()
    schema = gen.type_to_schema(Literal["hello"])  # type: ignore[valid-type]

    assert schema["const"] == "hello"
    assert schema["type"] == "string"


def test_multi_literal_generates_enum_schema():
    gen = JsonSchemaGenerator()
    schema = gen.type_to_schema(Literal["a", "b", "c"])  # type: ignore[valid-type]

    assert schema["enum"] == ["a", "b", "c"]
    assert schema["type"] == "string"


def test_mixed_type_literal_raises():
    gen = JsonSchemaGenerator()
    with pytest.raises((ValueError, TypeError)):
        _ = gen.type_to_schema(Literal["x", 1])  # type: ignore[valid-type]
