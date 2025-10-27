# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel


class MCPProviderDataValidator(BaseModel):
    # mcp_endpoint => dict of headers to send
    mcp_headers: dict[str, dict[str, str]] | None = None


class MCPProviderConfig(BaseModel):
    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {}
