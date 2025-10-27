# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.utils.bedrock.config import BedrockBaseConfig
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class BedrockSafetyConfig(BedrockBaseConfig):
    pass
