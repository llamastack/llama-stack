# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Router registry for FastAPI routers.

This module provides a way to register FastAPI routers for APIs that have been
migrated to use explicit FastAPI routers instead of Protocol-based route discovery.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter

if TYPE_CHECKING:
    from llama_stack.apis.datatypes import Api

# Registry of router factory functions
# Each factory function takes a callable that returns the implementation for a given API
# and returns an APIRouter
# Use string keys to avoid circular imports
_router_factories: dict[str, Callable[[Callable[[str], Any]], APIRouter]] = {}


def register_router(api: "Api", router_factory: Callable[[Callable[["Api"], Any]], APIRouter]) -> None:
    """Register a router factory for an API.

    Args:
        api: The API enum value
        router_factory: A function that takes an impl_getter function and returns an APIRouter
    """
    _router_factories[api.value] = router_factory  # type: ignore[attr-defined]


def has_router(api: "Api") -> bool:
    """Check if an API has a registered router.

    Args:
        api: The API enum value

    Returns:
        True if a router factory is registered for this API
    """
    return api.value in _router_factories  # type: ignore[attr-defined]


def create_router(api: "Api", impl_getter: Callable[["Api"], Any]) -> APIRouter | None:
    """Create a router for an API if one is registered.

    Args:
        api: The API enum value
        impl_getter: Function that returns the implementation for a given API

    Returns:
        APIRouter if registered, None otherwise
    """
    api_value = api.value  # type: ignore[attr-defined]
    if api_value not in _router_factories:
        return None
    return _router_factories[api_value](impl_getter)
