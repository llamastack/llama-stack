# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import aiohttp
from openai import NOT_GIVEN

from llama_stack.apis.inference import (
    RerankData,
    RerankResponse,
)
from llama_stack.apis.inference.inference import (
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
)
from llama_stack.apis.models import Model, ModelType
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from . import NVIDIAConfig
from .utils import _is_nvidia_hosted

logger = get_logger(name=__name__, category="inference::nvidia")


class NVIDIAInferenceAdapter(OpenAIMixin):
    config: NVIDIAConfig

    """
    NVIDIA Inference Adapter for Llama Stack.

    Note: The inheritance order is important here. OpenAIMixin must come before
    ModelRegistryHelper to ensure that OpenAIMixin.check_model_availability()
    is used instead of ModelRegistryHelper.check_model_availability(). It also
    must come before Inference to ensure that OpenAIMixin methods are available
    in the Inference interface.

    - OpenAIMixin.check_model_availability() queries the NVIDIA API to check if a model exists
    - ModelRegistryHelper.check_model_availability() just returns False and shows a warning
    """

    # source: https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/support-matrix.html
    embedding_model_metadata: dict[str, dict[str, int]] = {
        "nvidia/llama-3.2-nv-embedqa-1b-v2": {"embedding_dimension": 2048, "context_length": 8192},
        "nvidia/nv-embedqa-e5-v5": {"embedding_dimension": 512, "context_length": 1024},
        "nvidia/nv-embedqa-mistral-7b-v2": {"embedding_dimension": 512, "context_length": 4096},
        "snowflake/arctic-embed-l": {"embedding_dimension": 512, "context_length": 1024},
    }

    rerank_model_list = [
        "nv-rerank-qa-mistral-4b:1",
        "nvidia/nv-rerankqa-mistral-4b-v3",
        "nvidia/llama-3.2-nv-rerankqa-1b-v2",
    ]

    _rerank_model_endpoints = {
        "nv-rerank-qa-mistral-4b:1": "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
        "nvidia/nv-rerankqa-mistral-4b-v3": "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3/reranking",
        "nvidia/llama-3.2-nv-rerankqa-1b-v2": "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking",
    }

    async def initialize(self) -> None:
        logger.info(f"Initializing NVIDIAInferenceAdapter({self.config.url})...")

        if _is_nvidia_hosted(self.config):
            if not self.config.auth_credential:
                raise RuntimeError(
                    "API key is required for hosted NVIDIA NIM. Either provide an API key or use a self-hosted NIM."
                )

    def get_api_key(self) -> str:
        """
        Get the API key for OpenAI mixin.

        :return: The NVIDIA API key
        """
        if self.config.auth_credential:
            return self.config.auth_credential.get_secret_value()

        if not _is_nvidia_hosted(self.config):
            return "NO KEY REQUIRED"

        return None

    def get_base_url(self) -> str:
        """
        Get the base URL for OpenAI mixin.

        :return: The NVIDIA API base URL
        """
        return f"{self.config.url}/v1" if self.config.append_api_version else self.config.url

    async def list_models(self) -> list[Model] | None:
        """
        List available NVIDIA models by combining:
        1. Dynamic models from https://integrate.api.nvidia.com/v1/models
        2. Static rerank models (which use different API endpoints)
        """
        self._model_cache = {}
        models = await super().list_models()

        # Add rerank models
        existing_ids = {m.identifier for m in models}
        for model_id, _ in self._rerank_model_endpoints.items():
            if self.allowed_models and model_id not in self.allowed_models:
                continue
            if model_id not in existing_ids:
                model = Model(
                    provider_id=self.__provider_id__,  # type: ignore[attr-defined]
                    provider_resource_id=model_id,
                    identifier=model_id,
                    model_type=ModelType.rerank,
                )
                models.append(model)
                self._model_cache[model_id] = model

        return models

    async def rerank(
        self,
        model: str,
        query: str | OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam,
        items: list[str | OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam],
        max_num_results: int | None = None,
    ) -> RerankResponse:
        provider_model_id = await self._get_provider_model_id(model)

        ranking_url = self.get_base_url()

        if _is_nvidia_hosted(self._config) and provider_model_id in self._rerank_model_endpoints:
            ranking_url = self._rerank_model_endpoints[provider_model_id]

        logger.debug(f"Using rerank endpoint: {ranking_url} for model: {provider_model_id}")

        # Convert query to text format
        if isinstance(query, str):
            query_text = query
        elif isinstance(query, OpenAIChatCompletionContentPartTextParam):
            query_text = query.text
        else:
            raise ValueError("Query must be a string or text content part")

        # Convert items to text format
        passages = []
        for item in items:
            if isinstance(item, str):
                passages.append({"text": item})
            elif isinstance(item, OpenAIChatCompletionContentPartTextParam):
                passages.append({"text": item.text})
            else:
                raise ValueError("Items must be strings or text content parts")

        payload = {
            "model": provider_model_id,
            "query": {"text": query_text},
            "passages": passages,
        }

        headers = {
            "Authorization": f"Bearer {self.get_api_key()}",
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(ranking_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        raise ConnectionError(
                            f"NVIDIA rerank API request failed with status {response.status}: {response_text}"
                        )

                    result = await response.json()
                    rankings = result.get("rankings", [])

                    # Convert to RerankData format
                    rerank_data = []
                    for ranking in rankings:
                        rerank_data.append(RerankData(index=ranking["index"], relevance_score=ranking["logit"]))

                    # Apply max_num_results limit
                    if max_num_results is not None:
                        rerank_data = rerank_data[:max_num_results]

                    return RerankResponse(data=rerank_data)

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to NVIDIA rerank API at {ranking_url}: {e}") from e