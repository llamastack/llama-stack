# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Iterable
from typing import Any

from databricks.sdk import WorkspaceClient

from llama_stack.apis.inference import OpenAICompletion
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import DatabricksImplConfig

logger = get_logger(name=__name__, category="inference::databricks")


class DatabricksInferenceAdapter(OpenAIMixin):
    config: DatabricksImplConfig

    # source: https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models
    embedding_model_metadata: dict[str, dict[str, int]] = {
        "databricks-gte-large-en": {"embedding_dimension": 1024, "context_length": 8192},
        "databricks-bge-large-en": {"embedding_dimension": 1024, "context_length": 512},
    }

    def get_api_key(self) -> str:
        return self.config.api_token.get_secret_value()

    def get_base_url(self) -> str:
        return f"{self.config.url}/serving-endpoints"

    async def list_provider_model_ids(self) -> Iterable[str]:
        return [
            endpoint.name
            for endpoint in WorkspaceClient(
                host=self.config.url, token=self.get_api_key()
            ).serving_endpoints.list()  # TODO: this is not async
        ]

    async def should_refresh_models(self) -> bool:
        return False

    async def openai_completion(
        self,
        model: str,
        prompt: str | list[str] | list[int] | list[list[int]],
        best_of: int | None = None,
        echo: bool | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream: bool | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        user: str | None = None,
        guided_choice: list[str] | None = None,
        prompt_logprobs: int | None = None,
        suffix: str | None = None,
    ) -> OpenAICompletion:
        raise NotImplementedError()
