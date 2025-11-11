# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

from opentelemetry import trace

from llama_stack.apis.inference import Message
from llama_stack.apis.safety import RunShieldResponse

from .constants import (
    SAFETY_REQUEST_MESSAGES_ATTRIBUTE,
    SAFETY_REQUEST_SHIELD_ID_ATTRIBUTE,
    SAFETY_RESPONSE_METADATA_ATTRIBUTE,
    SAFETY_RESPONSE_USER_MESSAGE_ATTRIBUTE,
    SAFETY_RESPONSE_VIOLATION_LEVEL_ATTRIBUTE,
)


def guardrail_request_span_attributes(shield_id: str, messages: list[Message], response: RunShieldResponse) -> None:
    span = trace.get_current_span()
    if span is not None:
        span.set_attribute(SAFETY_REQUEST_SHIELD_ID_ATTRIBUTE, shield_id)
        messages_json = json.dumps([msg.model_dump() for msg in messages])
        span.set_attribute(SAFETY_REQUEST_MESSAGES_ATTRIBUTE, messages_json)

        if response.violation:
            if response.violation.metadata:
                metadata_json = json.dumps(response.violation.metadata)
                span.set_attribute(SAFETY_RESPONSE_METADATA_ATTRIBUTE, metadata_json)
            if response.violation.user_message:
                span.set_attribute(SAFETY_RESPONSE_USER_MESSAGE_ATTRIBUTE, response.violation.user_message)
            span.set_attribute(SAFETY_RESPONSE_VIOLATION_LEVEL_ATTRIBUTE, response.violation.violation_level.value)
