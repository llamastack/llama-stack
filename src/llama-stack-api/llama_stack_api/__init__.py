# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Llama Stack API Specifications

This package contains the API definitions, data types, and protocol specifications
for Llama Stack. It is designed to be a lightweight dependency for external providers
and clients that need to interact with Llama Stack APIs without requiring the full
server implementation.

Key components:
- API modules (agents, inference, safety, etc.): Protocol definitions for all Llama Stack APIs
- datatypes: Core data types and provider specifications
- common: Common data types used across APIs
- strong_typing: Type system utilities
- schema_utils: Schema validation and utilities
"""

__version__ = "0.1.0"

from . import common, datatypes, schema_utils, strong_typing  # noqa: F401

__all__ = ["common", "datatypes", "schema_utils", "strong_typing"]
