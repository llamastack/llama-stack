# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

from llama_stack.log import get_logger
from llama_stack_api import OpenAIMessageParam, Safety, SafetyViolation, ViolationLevel

log = get_logger(name=__name__, category="agents::meta_reference")


class SafetyException(Exception):  # noqa: N818
    def __init__(self, violation: SafetyViolation):
        self.violation = violation
        super().__init__(violation.user_message)


class ShieldRunnerMixin:
    def __init__(
        self,
        safety_api: Safety,
        input_shields: list[str] | None = None,
        output_shields: list[str] | None = None,
    ):
        self.safety_api = safety_api
        self.input_shields = input_shields
        self.output_shields = output_shields

    async def run_multiple_shields(self, messages: list[OpenAIMessageParam], identifiers: list[str]) -> None:
        responses = await asyncio.gather(
            *[
                self.safety_api.run_shield(shield_id=identifier, messages=messages, params={})
                for identifier in identifiers
            ]
        )
        for identifier, response in zip(identifiers, responses, strict=False):
            if not response.violation:
                continue

            violation = response.violation
            if violation.violation_level == ViolationLevel.ERROR:
                raise SafetyException(violation)
            elif violation.violation_level == ViolationLevel.WARN:
                log.warning(f"[Warn]{identifier} raised a warning")
