# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, field_validator

from llama_stack_api import json_schema_type

_BLOCKED_HEADERS = frozenset(
    {
        "host",
        "content-type",
        "content-length",
        "transfer-encoding",
        "connection",
        "upgrade",
        "te",
        "trailer",
        "cookie",
        "set-cookie",
    }
)


class PassthroughProviderDataValidator(BaseModel):
    # allow arbitrary keys so forward_headers can access them
    model_config = ConfigDict(extra="allow")

    passthrough_api_key: SecretStr | None = Field(
        default=None,
        description="API key for the downstream safety service",
    )


@json_schema_type
class PassthroughSafetyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: HttpUrl = Field(
        description="Base URL of the downstream safety service (e.g. https://safety.example.com/v1)",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="API key for the downstream safety service. If set, takes precedence over provider data.",
    )
    forward_headers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping of provider data keys to outbound HTTP header names. "
            "Only keys listed here are forwarded from X-LlamaStack-Provider-Data "
            'to the downstream service. Example: {"maas_api_token": "Authorization"}'
        ),
    )

    @field_validator("forward_headers")
    @classmethod
    def validate_forward_headers(cls, v: dict[str, str]) -> dict[str, str]:
        errors: list[str] = []
        for provider_key, header_name in v.items():
            if provider_key.startswith("__"):
                errors.append(f"provider key '{provider_key}' uses reserved __ prefix")
            if header_name.lower() in _BLOCKED_HEADERS:
                errors.append(f"header '{header_name}' is blocked (security-sensitive)")
        if errors:
            raise ValueError(f"invalid forward_headers: {'; '.join(errors)}")
        return v

    @classmethod
    def sample_run_config(
        cls,
        base_url: str = "${env.PASSTHROUGH_SAFETY_URL}",
        api_key: str = "${env.PASSTHROUGH_SAFETY_API_KEY:=}",
        forward_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "base_url": base_url,
            "api_key": api_key,
        }
        if forward_headers:
            config["forward_headers"] = forward_headers
        return config
