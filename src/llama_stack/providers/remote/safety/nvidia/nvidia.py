# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

import requests

from llama_stack.log import get_logger
from llama_stack_api import (
    ModerationObject,
    OpenAIMessageParam,
    RunShieldResponse,
    Safety,
    SafetyViolation,
    Shield,
    ShieldsProtocolPrivate,
    ViolationLevel,
)

from .config import NVIDIASafetyConfig

logger = get_logger(name=__name__, category="safety::nvidia")


class NVIDIASafetyAdapter(Safety, ShieldsProtocolPrivate):
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

        self.shield = NeMoGuardrails(self.config, model)
        return await self.shield.run(messages)

    async def run_moderation(self, input: str | list[str], model: str | None = None) -> ModerationObject:
        raise NotImplementedError("NVIDIA safety provider currently does not implement run_moderation")


class NeMoGuardrails:
    """
    A class that encapsulates NVIDIA's guardrails safety logic.

    Sends messages to the guardrails service and interprets the response to determine
    if a safety violation has occurred.
    """

    def __init__(
        self,
        config: NVIDIASafetyConfig,
        model: str,
        threshold: float = 0.9,
        temperature: float = 1.0,
    ):
        """
        Initialize a NeMoGuardrails instance with the provided parameters.

        Args:
            config (NVIDIASafetyConfig): The safety configuration containing the config ID and guardrails URL.
            model (str): The identifier or name of the model to be used for safety checks.
            threshold (float, optional): The threshold for flagging violations. Defaults to 0.9.
            temperature (float, optional): The temperature setting for the underlying model. Must be greater than 0. Defaults to 1.0.

        Raises:
            ValueError: If temperature is less than or equal to 0.
            AssertionError: If config_id is not provided in the configuration.
        """
        self.config_id = config.config_id
        self.model = model
        assert self.config_id is not None, "Must provide config id"
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")

        self.temperature = temperature
        self.threshold = threshold
        self.guardrails_service_url = config.guardrails_service_url

    async def _guardrails_post(self, path: str, data: Any | None):
        """Helper for making POST requests to the guardrails service."""
        headers = {
            "Accept": "application/json",
        }
        response = requests.post(url=f"{self.guardrails_service_url}{path}", headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    async def run(self, messages: list[OpenAIMessageParam]) -> RunShieldResponse:
        """
        Queries the /v1/guardrail/chat/completions endpoint of the NeMo guardrails deployed API.

        Args:
            messages (List[Message]): A list of Message objects to be checked for safety violations.

        Returns:
            RunShieldResponse: Response with SafetyViolation if content is blocked, None otherwise.

        Raises:
            requests.HTTPError: If the POST request fails.
        """
        request_data = {
            "config_id": self.config_id,
            "model": self.model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
        }
        response = await self._guardrails_post(path="/v1/guardrail/chat/completions", data=request_data)

        # Detection method 1: Check for error object with guardrails_violation
        # NeMo Guardrails returns this format when content is blocked:
        # {"error": {"message": "Blocked by <rail-name> rails.", "type": "guardrails_violation",
        #            "param": "<rail-name>", "code": "content_blocked"}}
        error = response.get("error")
        if error:
            error_type = error.get("type", "")
            error_code = error.get("code", "")
            if error_type == "guardrails_violation" or error_code == "content_blocked":
                return RunShieldResponse(
                    violation=SafetyViolation(
                        user_message=error.get("message", "Content blocked by guardrails"),
                        violation_level=ViolationLevel.ERROR,
                        metadata={
                            "error_type": error_type,
                            "error_code": error_code,
                            "rail_name": error.get("param", "unknown"),
                            "detection_method": "error_object",
                        },
                    )
                )

        # Detection method 2: Check for exception message with InputRailException
        # When enable_rails_exceptions is True, NeMo returns: {"role": "exception", "type": "InputRailException"}
        response_messages = response.get("messages", [])
        for msg in response_messages:
            if msg.get("role") == "exception":
                exception_type = msg.get("type", "RailException")
                return RunShieldResponse(
                    violation=SafetyViolation(
                        user_message=msg.get("content", "Content blocked by guardrails"),
                        violation_level=ViolationLevel.ERROR,
                        metadata={
                            "exception_type": exception_type,
                            "detection_method": "exception_message",
                        },
                    )
                )

        # Detection method 3: Legacy format with explicit status field
        if "status" in response and response["status"] == "blocked":
            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message="Content blocked by guardrails",
                    violation_level=ViolationLevel.ERROR,
                    metadata=response.get("rails_status", {}),
                )
            )

        # No violation detected
        return RunShieldResponse(violation=None)
