# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import Any

from llama_stack.apis.inference import OpenAIMessageParam
from llama_stack.apis.safety import (
    RunShieldResponse,
    Safety,
    SafetyViolation,
    ViolationLevel,
)
from llama_stack.apis.shields import Shield
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from llama_stack.providers.utils.bedrock.client import create_bedrock_client

from .config import BedrockSafetyConfig

logger = get_logger(name=__name__, category="safety::bedrock")


class BedrockSafetyAdapter(Safety, ShieldsProtocolPrivate):
    def __init__(self, config: BedrockSafetyConfig) -> None:
        self.config = config
        self.registered_shields = []

    async def initialize(self) -> None:
        try:
            self.bedrock_runtime_client = create_bedrock_client(self.config)
            self.bedrock_client = create_bedrock_client(self.config, "bedrock")
        except Exception as e:
            raise RuntimeError("Error initializing BedrockSafetyAdapter") from e

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        response = self.bedrock_client.list_guardrails(
            guardrailIdentifier=shield.provider_resource_id,
        )
        if (
            not response["guardrails"]
            or len(response["guardrails"]) == 0
            or response["guardrails"][0]["version"] != shield.params["guardrailVersion"]
        ):
            raise ValueError(
                f"Shield {shield.provider_resource_id} with version {shield.params['guardrailVersion']} not found in Bedrock"
            )

    async def unregister_shield(self, identifier: str) -> None:
        pass

    async def run_shield(
        self, shield_id: str, messages: list[OpenAIMessageParam], params: dict[str, Any] = None
    ) -> RunShieldResponse:
        shield = await self.shield_store.get_shield(shield_id)
        if not shield:
            raise ValueError(f"Shield {shield_id} not found")

        """
        This is the implementation for the bedrock guardrails. The input to the guardrails is to be of this format
        ```content = [
            {
                "text": {
                    "text": "Is the AB503 Product a better investment than the S&P 500?"
                }
            }
        ]```
        Incoming messages contain content, role . For now we will extract the content and
        default the "qualifiers": ["query"]
        """

        shield_params = shield.params
        logger.debug(f"run_shield::{shield_params}::messages={messages}")

        # - convert the messages into format Bedrock expects
        content_messages = []
        for message in messages:
            content_messages.append({"text": {"text": message.content}})
        logger.debug(f"run_shield::final:messages::{json.dumps(content_messages, indent=2)}:")

        response = self.bedrock_runtime_client.apply_guardrail(
            guardrailIdentifier=shield.provider_resource_id,
            guardrailVersion=shield_params["guardrailVersion"],
            source="OUTPUT",  # or 'INPUT' depending on your use case
            content=content_messages,
        )
        if response["action"] == "GUARDRAIL_INTERVENED":
            user_message = ""
            metadata = {}
            for output in response["outputs"]:
                # guardrails returns a list - however for this implementation we will leverage the last values
                user_message = output["text"]
            for assessment in response["assessments"]:
                # guardrails returns a list - however for this implementation we will leverage the last values
                metadata = dict(assessment)

            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message=user_message,
                    violation_level=ViolationLevel.ERROR,
                    metadata=metadata,
                )
            )

        return RunShieldResponse()
