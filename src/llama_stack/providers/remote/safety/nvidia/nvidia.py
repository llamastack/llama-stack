# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

import httpx

from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.prompt_adapter import interleaved_content_as_str
from llama_stack_api import (
    ModerationObject,
    OpenAIMessageParam,
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


class NVIDIASafetyAdapter(Safety, ShieldsProtocolPrivate):
    shield_store: ShieldStore

    def __init__(self, config: NVIDIASafetyConfig) -> None:
        """
        Initialize the NVIDIASafetyAdapter with a given safety configuration.

        Args:
            config (NVIDIASafetyConfig): The configuration containing the guardrails service URL and config ID.
        """
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

    async def run_shield(
        self, shield_id: str, messages: list[OpenAIMessageParam], params: dict[str, Any] | None = None
    ) -> RunShieldResponse:
        """
        Run a safety shield check against the provided messages.

        Args:
            shield_id (str): The unique identifier for the shield to be used.
            messages (List[Message]): A list of Message objects representing the conversation history.
            params (Optional[dict[str, Any]]): Additional parameters for the shield check.

        Returns:
            RunShieldResponse: The response containing safety violation details if any.

        Raises:
            ValueError: If the shield with the provided shield_id is not found.
        """
        shield = await self.shield_store.get_shield(shield_id)
        if not shield:
            raise ValueError(f"Shield {shield_id} not found")

        shield_params = shield.params or {}
        model = shield_params.get("model") or shield.provider_resource_id
        if not model:
            raise ValueError(
                f"Shield {shield_id} does not have a model configured. Set 'model' in params or provider_resource_id."
            )

        guardrails = NeMoGuardrails(self.config, model)
        return await guardrails.run(messages)

    async def run_moderation(self, input: str | list[str], model: str | None = None) -> ModerationObject:
        raise NotImplementedError("NVIDIA safety provider currently does not implement run_moderation")


class NeMoGuardrails:
    """
    A class that encapsulates NVIDIA's guardrails safety logic.

    Sends messages to the guardrails service and interprets the response to determine
    if a safety violation has occurred.
    """

    def __init__(self, config: NVIDIASafetyConfig, model: str):
        """
        Initialize a NeMoGuardrails instance with the provided parameters.

        Args:
            config (NVIDIASafetyConfig): The safety configuration containing the config ID and guardrails URL.
            model (str): The identifier or name of the model to be used for safety checks.
        """
        self.config_id = config.config_id
        self.model = model
        self.blocked_message = config.blocked_message
        self.guardrails_service_url = config.guardrails_service_url
        self.temperature = config.temperature
        self.timeout = config.timeout
        self.api_mode = config.api_mode

    async def _guardrails_post(self, path: str, data: Any | None) -> dict[str, Any]:
        """Make a POST request to the guardrails service."""
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        url = f"{self.guardrails_service_url}{path}"
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
        """
        Run safety check against the NeMo Guardrails API.

        Args:
            messages (List[Message]): A list of Message objects to be checked for safety violations.

        Returns:
            RunShieldResponse: Response with SafetyViolation if content is blocked, None otherwise.

        Raises:
            httpx.HTTPStatusError: If the POST request fails.
        """
        if self.api_mode == GuardrailsApiMode.MICROSERVICE:
            return await self._run_microservice(messages)
        return await self._run_openai(messages)

    async def _run_microservice(self, messages: list[OpenAIMessageParam]) -> RunShieldResponse:
        """Enterprise NIM endpoint: /v1/guardrail/checks"""
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
            f"Guardrails request (microservice) to {self.guardrails_service_url}: config_id={self.config_id}, model={self.model}"
        )
        response = await self._guardrails_post(path="/v1/guardrail/checks", data=request_data)

        if response.get("status") == "blocked":
            logger.info(f"Guardrails blocked: {response.get('rails_status', {})}")
            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message="Sorry I cannot do this.",
                    violation_level=ViolationLevel.ERROR,
                    metadata=response.get("rails_status", {}),
                )
            )
        return RunShieldResponse(violation=None)

    async def _run_openai(self, messages: list[OpenAIMessageParam]) -> RunShieldResponse:
        """Open-source toolkit endpoint: /v1/guardrail/chat/completions"""
        request_data = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": interleaved_content_as_str(message.content)} for message in messages
            ],
            "guardrails": {"config_id": self.config_id},
            "temperature": self.temperature,
        }
        logger.debug(
            f"Guardrails request (openai) to {self.guardrails_service_url}: config_id={self.config_id}, model={self.model}"
        )
        response = await self._guardrails_post(path="/v1/guardrail/chat/completions", data=request_data)

        error = response.get("error")
        if error:
            error_type = error.get("type", "")
            error_code = error.get("code", "")
            if error_type == "guardrails_violation" or error_code == "content_blocked":
                logger.info(
                    f"Guardrails violation: type={error_type}, code={error_code}, rail={error.get('param', 'unknown')}"
                )
                return RunShieldResponse(
                    violation=SafetyViolation(
                        user_message=error.get("message", "Content blocked by guardrails"),
                        violation_level=ViolationLevel.ERROR,
                        metadata={
                            "error_type": error_type,
                            "error_code": error_code,
                            "rail_name": error.get("param", "unknown"),
                        },
                    )
                )
            # Unknown error type - log and raise to avoid silent failure
            logger.error(
                f"Guardrails service error: type={error_type}, code={error_code}, message={error.get('message', 'unknown')}"
            )
            raise RuntimeError(f"Failed to run guardrails check: {error.get('message', 'Unknown error')}")

        # Check for legacy format with status field
        if response.get("status") == "blocked":
            logger.info(f"Guardrails blocked (legacy format): {response.get('rails_status', {})}")
            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message="Content blocked by guardrails",
                    violation_level=ViolationLevel.ERROR,
                    metadata=response.get("rails_status", {}),
                )
            )

        # Extract response content - handle both OpenAI format (choices) and NeMo format (messages)
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
                            user_message=msg.get("content", "Content blocked by guardrails"),
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
