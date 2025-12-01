# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
This file contains constants used for naming data captured for telemetry.

This is used to ensure that the data captured for telemetry is consistent and can be used to
identify and correlate data. If custom telemetry data is added to llama stack, please add
constants for it here.
"""

llama_stack_prefix = "llama_stack"

# Safety Attributes
RUN_SHIELD_OPERATION_NAME = "run_shield"

SAFETY_REQUEST_PREFIX = f"{llama_stack_prefix}.safety.request"
SAFETY_REQUEST_SHIELD_ID_ATTRIBUTE = f"{SAFETY_REQUEST_PREFIX}.shield_id"
SAFETY_REQUEST_MESSAGES_ATTRIBUTE = f"{SAFETY_REQUEST_PREFIX}.messages"

SAFETY_RESPONSE_PREFIX = f"{llama_stack_prefix}.safety.response"
SAFETY_RESPONSE_METADATA_ATTRIBUTE = f"{SAFETY_RESPONSE_PREFIX}.metadata"
SAFETY_RESPONSE_VIOLATION_LEVEL_ATTRIBUTE = f"{SAFETY_RESPONSE_PREFIX}.violation.level"
SAFETY_RESPONSE_USER_MESSAGE_ATTRIBUTE = f"{SAFETY_RESPONSE_PREFIX}.violation.user_message"
