# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import uuid
from typing import Any

import httpx

from ogx.core.request_headers import NeedsRequestProviderData
from ogx.log import get_logger
from ogx.providers.utils.forward_headers import build_forwarded_headers
from ogx_api import (
    ModerationObject,
    ModerationObjectResults,
    RunModerationRequest,
    Safety,
    Shield,
    ShieldsProtocolPrivate,
)

from .config import PassthroughSafetyConfig

logger = get_logger(__name__, category="safety")


class PassthroughSafetyAdapter(
    Safety,
    ShieldsProtocolPrivate,
    NeedsRequestProviderData,
):
    """Forwards moderation calls to a downstream service via /v1/moderations."""

    shield_store: Any  # injected by framework after initialization

    def __init__(self, config: PassthroughSafetyConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        await asyncio.shield(self._client.aclose())

    async def register_shield(self, shield: Shield) -> None:
        pass

    async def unregister_shield(self, identifier: str) -> None:
        pass

    def _get_api_key(self) -> str | None:
        if self.config.api_key is not None:
            value = self.config.api_key.get_secret_value()
            if value:
                return value

        provider_data = self.get_request_provider_data()
        if provider_data is not None and provider_data.passthrough_api_key:
            return str(provider_data.passthrough_api_key.get_secret_value())

        return None

    def _build_forward_headers(self) -> dict[str, str]:
        provider_data = self.get_request_provider_data()
        forwarded = build_forwarded_headers(provider_data, self.config.forward_headers)
        if self.config.forward_headers and not forwarded:
            logger.warning(
                "forward_headers is configured but no matching keys found in provider data — "
                "outbound request may be unauthenticated"
            )
        return forwarded

    def _build_request_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        headers.update(self._build_forward_headers())
        api_key = self._get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def run_moderation(self, request: RunModerationRequest) -> ModerationObject:
        """Forward directly to downstream /v1/moderations."""
        inputs = request.input if isinstance(request.input, list) else [request.input]

        payload: dict[str, str | list[str]] = {"input": request.input}
        if request.model is not None:
            payload["model"] = request.model

        base_url = str(self.config.base_url).rstrip("/")
        url = f"{base_url}/moderations"

        headers = self._build_request_headers()

        data = await self._post_moderation(url, payload, headers)

        results_data = data.get("results")
        if not isinstance(results_data, list):
            raise RuntimeError("Downstream safety service returned malformed response (missing or invalid 'results')")
        results: list[ModerationObjectResults] = []

        for result in results_data:
            if not isinstance(result, dict):
                raise RuntimeError("Downstream safety service returned malformed result entry (expected object)")
            flagged = result.get("flagged", False)
            categories = result.get("categories") or {}
            category_scores = result.get("category_scores") or {}

            results.append(
                ModerationObjectResults(
                    flagged=flagged,
                    categories=categories,
                    category_scores=category_scores,
                    category_applied_input_types=result.get("category_applied_input_types"),
                    user_message=None,
                    metadata={},
                )
            )

        if len(results) != len(inputs):
            raise RuntimeError(f"Downstream safety service returned {len(results)} results for {len(inputs)} inputs")

        return ModerationObject(
            id=data.get("id", f"modr-{uuid.uuid4()}"),
            model=data.get("model", request.model or ""),
            results=results,
        )

    async def _post_moderation(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        try:
            response = await self._client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.TimeoutException as e:
            raise RuntimeError("Failed to reach downstream safety service: request timed out") from e
        except httpx.ConnectError as e:
            raise RuntimeError("Failed to reach downstream safety service: connection failed") from e
        except httpx.HTTPStatusError as e:
            if 400 <= e.response.status_code < 500:
                raise ValueError(
                    f"Downstream safety service rejected the request (HTTP {e.response.status_code})"
                ) from e
            raise RuntimeError(f"Downstream safety service returned HTTP {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise RuntimeError("Failed to reach downstream safety service: unexpected request error") from e

        try:
            raw = response.json()
        except (ValueError, UnicodeDecodeError) as e:
            raise RuntimeError(
                f"Downstream safety service returned non-JSON response (HTTP {response.status_code})"
            ) from e

        if not isinstance(raw, dict):
            raise RuntimeError("Downstream safety service returned invalid response (expected JSON object)")

        return raw
