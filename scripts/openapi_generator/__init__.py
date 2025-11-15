# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
OpenAPI generator module for Llama Stack.

This module provides functionality to generate OpenAPI specifications
from FastAPI applications.
"""

from .main import generate_openapi_spec, main

__all__ = ["generate_openapi_spec", "main"]
