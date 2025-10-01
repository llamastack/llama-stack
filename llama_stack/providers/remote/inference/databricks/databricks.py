# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from typing import Any

from databricks.sdk import WorkspaceClient

from llama_stack.apis.inference import (
    ChatCompletionResponse,
    ChatCompletionResponseStreamChunk,
    Inference,
    LogProbConfig,
    Message,
    Model,
    OpenAICompletion,
    ResponseFormat,
    SamplingParams,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.apis.models import ModelType
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import DatabricksImplConfig

logger = get_logger(name=__name__, category="inference::databricks")


class DatabricksInferenceAdapter(
    OpenAIMixin,
    Inference,
):
    # source: https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models
    embedding_model_metadata = {
        "databricks-gte-large-en": {"embedding_dimension": 1024, "context_length": 8192},
        "databricks-bge-large-en": {"embedding_dimension": 1024, "context_length": 512},
    }

    def __init__(self, config: DatabricksImplConfig) -> None:
        self.config = config

    def get_api_key(self) -> str:
        return self.config.api_token.get_secret_value()

    def get_base_url(self) -> str:
        return f"{self.config.url}/serving-endpoints"

    async def initialize(self) -> None:
        return

    async def shutdown(self) -> None:
        pass

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

    async def chat_completion(
        self,
        model_id: str,
        messages: list[Message],
        sampling_params: SamplingParams | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = ToolChoice.auto,
        tool_prompt_format: ToolPromptFormat | None = None,
        response_format: ResponseFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
        tool_config: ToolConfig | None = None,
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionResponseStreamChunk]:
        raise NotImplementedError()

    async def list_models(self) -> list[Model] | None:
        self._model_cache = {}  # from OpenAIMixin
        ws_client = WorkspaceClient(host=self.config.url, token=self.get_api_key())  # TODO: this is not async
        endpoints = ws_client.serving_endpoints.list()
        for endpoint in endpoints:
            model = Model(
                provider_id=self.__provider_id__,
                provider_resource_id=endpoint.name,
                identifier=endpoint.name,
            )
            if endpoint.task == "llm/v1/chat":
                model.model_type = ModelType.llm  # this is redundant, but informative
            elif endpoint.task == "llm/v1/embeddings":
                if endpoint.name not in self.embedding_model_metadata:
                    logger.warning(f"No metadata information available for embedding model {endpoint.name}, skipping.")
                    continue
                model.model_type = ModelType.embedding
                model.metadata = self.embedding_model_metadata[endpoint.name]
            else:
                logger.warning(f"Unknown model type, skipping: {endpoint}")
                continue

            self._model_cache[endpoint.name] = model

        return list(self._model_cache.values())

    async def should_refresh_models(self) -> bool:
        return False
