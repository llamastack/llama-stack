# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.remote.inference.llama_cpp_server.config import LlamaCppServerConfig
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin


class LlamaCppServerInferenceAdapter(OpenAIMixin):
    config: LlamaCppServerConfig

    def get_api_key(self) -> str | None:
        if self.config.auth_credential is None:
            return "NO KEY REQUIRED"
        return self.config.auth_credential.get_secret_value()

    def get_base_url(self) -> str:
        return str(self.config.base_url)
