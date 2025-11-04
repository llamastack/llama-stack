# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from collections.abc import Iterable

import aiohttp

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

    provider_data_api_key_field: str = "nvidia_api_key"

    """
    NVIDIA Inference Adapter for Llama Stack.
    """

    # source: https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/support-matrix.html
    embedding_model_metadata: dict[str, dict[str, int]] = {
        "nvidia/llama-3.2-nv-embedqa-1b-v2": {"embedding_dimension": 2048, "context_length": 8192},
        "nvidia/nv-embedqa-e5-v5": {"embedding_dimension": 512, "context_length": 1024},
        "nvidia/nv-embedqa-mistral-7b-v2": {"embedding_dimension": 512, "context_length": 4096},
        "snowflake/arctic-embed-l": {"embedding_dimension": 512, "context_length": 1024},
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

    async def list_provider_model_ids(self) -> Iterable[str]:
        """
        Return both dynamic model IDs and statically configured rerank model IDs.
        """
        dynamic_ids: Iterable[str] = []
        try:
            dynamic_ids = await super().list_provider_model_ids()
        except Exception:
            # If the dynamic listing fails, proceed with just configured rerank IDs
            dynamic_ids = []

        configured_rerank_ids = list(self.config.rerank_model_to_url.keys())
        return list(dict.fromkeys(list(dynamic_ids) + configured_rerank_ids))  # remove duplicates

    def construct_model_from_identifier(self, identifier: str) -> Model:
        """
        Classify rerank models from config; otherwise use the base behavior.
        """
        if identifier in self.config.rerank_model_to_url:
            return Model(
                provider_id=self.__provider_id__,  # type: ignore[attr-defined]
                provider_resource_id=identifier,
                identifier=identifier,
                model_type=ModelType.rerank,
            )
        return super().construct_model_from_identifier(identifier)

    async def rerank(
        self,
        model: str,
        query: str | OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam,
        items: list[str | OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam],
        max_num_results: int | None = None,
    ) -> RerankResponse:
        provider_model_id = await self._get_provider_model_id(model)

        ranking_url = self.get_base_url()

        if _is_nvidia_hosted(self.config) and provider_model_id in self.config.rerank_model_to_url:
            ranking_url = self.config.rerank_model_to_url[provider_model_id]

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
