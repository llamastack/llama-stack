# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Shared state for the OpenAPI generator module.
"""

from typing import Any

from llama_stack.apis.datatypes import Api

# Global list to store dynamic models created during endpoint generation
_dynamic_models: list[Any] = []

# Cache for protocol methods to avoid repeated lookups
_protocol_methods_cache: dict[Api, dict[str, Any]] | None = None

# Global dict to store extra body field information by endpoint
# Key: (path, method) tuple, Value: list of (param_name, param_type, description) tuples
_extra_body_fields: dict[tuple[str, str], list[tuple[str, type, str | None]]] = {}
