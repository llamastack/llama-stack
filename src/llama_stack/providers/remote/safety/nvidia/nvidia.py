# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any
from urllib.parse import urljoin

import httpx

from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.prompt_adapter import interleaved_content_as_str
from llama_stack.providers.utils.safety import ShieldToModerationMixin
from llama_stack_api import (
    GetShieldRequest,
    OpenAIMessageParam,
    RunShieldRequest,
    RunShieldResponse,
    Safety,
    SafetyViolation,
    Shield,
    ShieldsProtocolPrivate,
    ShieldStore,
    ViolationLevel,
)

from .config import GuardrailsApiMode, NVIDIASafetyConfig

logger = get_logger(name=__name__, category="safety::nvidia")


class NVIDIASafetyAdapter(ShieldToModerationMixin, Safety, ShieldsProtocolPrivate):
    shield_store: ShieldStore

    def __init__(self, config: NVIDIASafetyConfig) -> None:
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        if not shield.provider_resource_id:
            raise ValueError("Shield model not provided.")

    async def unregister_shield(self, identifier: str) -> None:
        pass

    async def run_shield(self, request: RunShieldRequest) -> RunShieldResponse:
        shield = await self.shield_store.get_shield(GetShieldRequest(identifier=request.shield_id))
        if not shield:
            raise ValueError(f"Shield {request.shield_id} not found")

        shield_params = shield.params or {}
        model = shield_params.get("model") or shield.provider_resource_id
        if not model:
            raise ValueError(
                f"Failed to run shield {request.shield_id}: no model configured. "
                "Set 'model' in params or provider_resource_id."
            )

        guardrails = NeMoGuardrails(self.config, model)
        return await guardrails.run(request.messages)


class NeMoGuardrails:
    def __init__(self, config: NVIDIASafetyConfig, model: str):
        self.config_id = config.config_id
        self.model = model
        self.blocked_message = config.blocked_message
        self.guardrails_service_url = config.guardrails_service_url
        self.temperature = config.temperature
        self.timeout = config.timeout
        self.api_mode = config.api_mode

    async def _guardrails_post(self, path: str, data: Any | None) -> dict[str, Any]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        url = urljoin(self.guardrails_service_url, path)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url=url, headers=headers, json=data)
                response.raise_for_status()
                return dict(response.json())
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"Failed to get response from guardrails service: request timed out after {self.timeout}s"
            ) from e
        except httpx.HTTPStatusError as e:
            error_text = e.response.text[:200] if e.response.text else "No response body"
            raise RuntimeError(f"Failed to call guardrails service: {e.response.status_code} {error_text}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to connect to guardrails service at {url}: {e}") from e

    async def run(self, messages: list[OpenAIMessageParam]) -> RunShieldResponse:
        if self.api_mode == GuardrailsApiMode.GUARDRAIL_CHECKS:
            return await self._run_guardrail_checks(messages)
        return await self._run_guardrail_chat_completions(messages)

    async def _run_guardrail_checks(self, messages: list[OpenAIMessageParam]) -> RunShieldResponse:
        request_data = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": interleaved_content_as_str(message.content)} for message in messages
            ],
            "temperature": self.temperature,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 160,
            "stream": False,
            "guardrails": {"config_id": self.config_id},
        }
        logger.debug(
            f"Guardrails request (guardrail_checks) to {self.guardrails_service_url}: config_id={self.config_id}, model={self.model}"
        )
        response = await self._guardrails_post(path="/v1/guardrail/checks", data=request_data)

        # Primary: check guardrails_data.log.activated_rails for any rail with stop=true
        activated_rails = response.get("guardrails_data", {}).get("log", {}).get("activated_rails", [])
        blocking_rails = [r for r in activated_rails if r.get("stop")]
        if blocking_rails:
            rail_names = [r.get("name", "unknown") for r in blocking_rails]
            logger.info(f"Guardrails blocked by rails: {rail_names}")
            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message=self.blocked_message,
                    violation_level=ViolationLevel.ERROR,
                    metadata={"blocking_rails": rail_names},
                )
            )

        # Fallback: legacy status field
        if response.get("status") == "blocked":
            logger.info(f"Guardrails blocked (legacy): {response.get('rails_status', {})}")
            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message=self.blocked_message,
                    violation_level=ViolationLevel.ERROR,
                    metadata=response.get("rails_status", {}),
                )
            )
        return RunShieldResponse(violation=None)

    async def _run_guardrail_chat_completions(self, messages: list[OpenAIMessageParam]) -> RunShieldResponse:
        request_data = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": interleaved_content_as_str(message.content)} for message in messages
            ],
            "guardrails": {"config_id": self.config_id},
            "temperature": self.temperature,
        }
        logger.debug(
            f"Guardrails request (guardrail_chat_completions) to {self.guardrails_service_url}: config_id={self.config_id}, model={self.model}"
        )
        response = await self._guardrails_post(path="/v1/guardrail/chat/completions", data=request_data)
        return self._parse_guardrails_response(response)

    def _parse_guardrails_response(self, response: dict[str, Any]) -> RunShieldResponse:
        error = response.get("error")
        if error:
            return self._handle_error_response(error)

        # Check for legacy format with status field
        if response.get("status") == "blocked":
            logger.info(f"Guardrails blocked (legacy format): {response.get('rails_status', {})}")
            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message=self.blocked_message,
                    violation_level=ViolationLevel.ERROR,
                    metadata=response.get("rails_status", {}),
                )
            )

        return self._parse_response_content(response)

    def _handle_error_response(self, error: dict[str, Any]) -> RunShieldResponse:
        error_type = error.get("type", "")
        error_code = error.get("code", "")

        if error_type == "guardrails_violation" or error_code == "content_blocked":
            logger.info(
                f"Guardrails violation: type={error_type}, code={error_code}, rail={error.get('param', 'unknown')}"
            )
            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message=error.get("message", self.blocked_message),
                    violation_level=ViolationLevel.ERROR,
                    metadata={
                        "error_type": error_type,
                        "error_code": error_code,
                        "rail_name": error.get("param", "unknown"),
                    },
                )
            )

        logger.error(
            f"Guardrails service error: type={error_type}, code={error_code}, message={error.get('message', 'unknown')}"
        )
        raise RuntimeError(f"Failed to run guardrails check: {error.get('message', 'Unknown error')}")

    def _parse_response_content(self, response: dict[str, Any]) -> RunShieldResponse:
        content = ""
        choices = response.get("choices", [])

        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content", "")
        else:
            for msg in response.get("messages", []):
                role = msg.get("role")
                if role == "exception":
                    return RunShieldResponse(
                        violation=SafetyViolation(
                            user_message=msg.get("content", self.blocked_message),
                            violation_level=ViolationLevel.ERROR,
                            metadata={"exception_type": msg.get("type", "RailException")},
                        )
                    )
                if role == "assistant":
                    content = msg.get("content", "")
                    break

        if content and self.blocked_message and self.blocked_message.lower().strip() == content.lower().strip():
            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message=content,
                    violation_level=ViolationLevel.ERROR,
                    metadata={"matched_pattern": self.blocked_message},
                )
            )

        return RunShieldResponse(violation=None)
