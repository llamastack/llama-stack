# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import Field, SecretStr

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaImplConfig(RemoteInferenceProviderConfig):
    auth_credential: SecretStr | None = Field(default=None, exclude=True)

    url: str = DEFAULT_OLLAMA_URL

    @classmethod
    def sample_run_config(cls, url: str = "${env.OLLAMA_URL:=http://localhost:11434}", **kwargs) -> dict[str, Any]:
        return {
            "url": url,
        }
