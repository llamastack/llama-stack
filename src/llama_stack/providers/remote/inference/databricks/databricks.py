# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Iterable

from databricks.sdk import WorkspaceClient

from llama_stack.apis.inference import OpenAICompletion, OpenAICompletionRequestWithExtraBody
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import DatabricksImplConfig

logger = get_logger(name=__name__, category="inference::databricks")


class DatabricksInferenceAdapter(OpenAIMixin):
    config: DatabricksImplConfig

    provider_data_api_key_field: str = "databricks_api_token"

    # source: https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models
    embedding_model_metadata: dict[str, dict[str, int]] = {
        "databricks-gte-large-en": {"embedding_dimension": 1024, "context_length": 8192},
        "databricks-bge-large-en": {"embedding_dimension": 1024, "context_length": 512},
    }

    def get_base_url(self) -> str:
        return f"{self.config.url}/serving-endpoints"

    async def list_provider_model_ids(self) -> Iterable[str]:
        # Filter out None values from endpoint names
        api_token = self._get_api_key_from_config_or_provider_data()
        return [
            endpoint.name  # type: ignore[misc]
            for endpoint in WorkspaceClient(
                host=self.config.url, token=api_token
            ).serving_endpoints.list()  # TODO: this is not async
        ]

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion:
        raise NotImplementedError()
