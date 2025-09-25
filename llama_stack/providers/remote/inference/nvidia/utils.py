# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import httpx

from llama_stack.log import get_logger

from . import NVIDIAConfig

logger = get_logger(name=__name__, category="inference::nvidia")


def _is_nvidia_hosted(config: NVIDIAConfig) -> bool:
    return "integrate.api.nvidia.com" in config.url


async def _get_health(url: str) -> tuple[bool, bool]:
    """
    Query {url}/v1/health/{live,ready} to check if the server is running and ready

    Args:
        url (str): URL of the server

    Returns:
        Tuple[bool, bool]: (is_live, is_ready)
    """
    async with httpx.AsyncClient() as client:
        live = await client.get(f"{url}/v1/health/live")
        ready = await client.get(f"{url}/v1/health/ready")
        return live.status_code == 200, ready.status_code == 200


async def check_health(config: NVIDIAConfig) -> None:
    """
    Check if the server is running and ready

    Args:
        url (str): URL of the server

    Raises:
        RuntimeError: If the server is not running or ready
    """
    if not _is_nvidia_hosted(config):
        logger.info("Checking NVIDIA NIM health...")
        try:
            is_live, is_ready = await _get_health(config.url)
            if not is_live:
                raise ConnectionError("NVIDIA NIM is not running")
            if not is_ready:
                raise ConnectionError("NVIDIA NIM is not ready")
            # TODO(mf): should we wait for the server to be ready?
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to NVIDIA NIM: {e}") from e
