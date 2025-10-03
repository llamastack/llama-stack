# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Mock server startup harness for parallel initialization.

HOW TO ADD A NEW MOCK SERVER:
1. Import your mock server class
2. Add it to MOCK_SERVERS list with configuration
3. Done! It will start in parallel with others.
"""

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from .mock_base import MockServerBase


class MockServerConfig(BaseModel):
    """
    Configuration for a mock server to start.

    **TO ADD A NEW MOCK SERVER:**
    Just create a MockServerConfig instance with your server class.

    Example:
        MockServerConfig(
            name="Mock MyService",
            server_class=MockMyService,
            init_kwargs={"port": 9000, "config_param": "value"},
        )
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str = Field(description="Display name for logging")
    server_class: type = Field(description="Mock server class (must inherit from MockServerBase)")
    init_kwargs: dict[str, Any] = Field(default_factory=dict, description="Kwargs to pass to server constructor")


async def start_mock_servers_async(mock_servers_config: list[MockServerConfig]) -> dict[str, MockServerBase]:
    """
    Start all mock servers in parallel and wait for them to be ready.

    **HOW IT WORKS:**
    1. Creates all server instances
    2. Calls await_start() on all servers in parallel
    3. Returns when all are ready

    **SIMPLE TO USE:**
        servers = await start_mock_servers_async([config1, config2, ...])

    Args:
        mock_servers_config: List of mock server configurations

    Returns:
        Dict mapping server name to server instance
    """
    servers = {}
    start_tasks = []

    # Create all servers and prepare start tasks
    for config in mock_servers_config:
        server = config.server_class(**config.init_kwargs)
        servers[config.name] = server
        start_tasks.append(server.await_start())

    # Start all servers in parallel
    try:
        await asyncio.gather(*start_tasks)

        # Print readiness confirmation
        for name in servers.keys():
            print(f"[INFO] {name} ready")

    except Exception as e:
        # If any server fails, stop all servers
        for server in servers.values():
            try:
                server.stop()
            except Exception:
                pass
        raise RuntimeError(f"Failed to start mock servers: {e}") from None

    return servers


def stop_mock_servers(servers: dict[str, Any]):
    """
    Stop all mock servers.

    Args:
        servers: Dict of server instances from start_mock_servers_async()
    """
    for name, server in servers.items():
        try:
            if hasattr(server, "get_request_count"):
                print(f"\n[INFO] {name} received {server.get_request_count()} requests")
            server.stop()
        except Exception as e:
            print(f"[WARN] Error stopping {name}: {e}")
