# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.remote.inference.groq.config import GroqConfig
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin


class GroqInferenceAdapter(OpenAIMixin):
    config: GroqConfig

    provider_data_api_key_field: str = "groq_api_key"

    def get_base_url(self) -> str:
        return f"{self.config.url}/openai/v1"
