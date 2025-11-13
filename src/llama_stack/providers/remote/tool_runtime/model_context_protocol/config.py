# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel


class MCPProviderDataValidator(BaseModel):
    """
    Validator for MCP provider-specific data passed via request headers.

    Note: MCP authentication and headers are now configured via the request body
    (OpenAIResponseInputToolMCP.authorization and .headers fields) rather than
    via provider data to simplify the API and avoid multiple configuration paths.

    This validator is kept for future provider-data extensions if needed.
    """

    pass


class MCPProviderConfig(BaseModel):
    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {}
