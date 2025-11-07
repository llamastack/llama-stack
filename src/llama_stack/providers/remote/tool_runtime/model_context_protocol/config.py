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
    Example usage:
        HTTP Request Headers:
            X-LlamaStack-Provider-Data: {
                "mcp_headers": {
                    "http://weather-mcp.com": {
                        "X-Trace-ID": "trace-123",
                        "X-Request-ID": "req-456"
                    }
                },
                "mcp_authorization": {
                    "http://weather-mcp.com": "weather_api_token_xyz"
                }
            }
    Security Note:
        - Authorization header MUST NOT be placed in mcp_headers
        - Use the dedicated mcp_authorization field instead
        - Each MCP endpoint can have its own separate token
    """

    # mcp_endpoint => dict of headers to send (excluding Authorization)
    mcp_headers: dict[str, dict[str, str]] | None = None

    # mcp_endpoint => authorization token
    # Example: {"http://server.com": "token123"}
    mcp_authorization: dict[str, str] | None = None


class MCPProviderConfig(BaseModel):
    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {}
