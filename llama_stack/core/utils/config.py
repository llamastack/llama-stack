# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any


def redact_sensitive_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive information from config before printing."""
    sensitive_patterns = ["api_key", "api_token", "password", "secret"]

    def _redact_value(v: Any) -> Any:
        if isinstance(v, dict):
            return _redact_dict(v)
        elif isinstance(v, list):
            return [_redact_value(i) for i in v]
        return v

    def _redact_dict(d: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for k, v in d.items():
            if any(pattern in k.lower() for pattern in sensitive_patterns):
                result[k] = "********"
            else:
                result[k] = _redact_value(v)
        return result

    return _redact_dict(data)
