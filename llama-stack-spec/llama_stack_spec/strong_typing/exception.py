# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Type-safe data interchange for Python data classes.

:see: https://github.com/hunyadi/strong_typing
"""


class JsonKeyError(Exception):
    "Raised when deserialization for a class or union type has failed because a matching member was not found."


class JsonValueError(Exception):
    "Raised when (de)serialization of data has failed due to invalid value."


class JsonTypeError(Exception):
    "Raised when deserialization of data has failed due to a type mismatch."
