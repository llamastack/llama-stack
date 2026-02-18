# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from collections.abc import AsyncIterator
from urllib.parse import urljoin

import aiohttp
import httpx
from pydantic import ConfigDict

from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin
from llama_stack_api import (
    HealthResponse,
    HealthStatus,
    Model,
    ModelType,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIChatCompletionRequestWithExtraBody,
    RerankData,
    RerankResponse,
    ToolChoice,
)
from llama_stack_api.inference import RerankRequest

from .config import VLLMInferenceAdapterConfig

log = get_logger(name=__name__, category="inference::vllm")


class VLLMInferenceAdapter(OpenAIMixin):
    config: VLLMInferenceAdapterConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # vLLM does not support the stream_options parameter
    supports_stream_options: bool = False

    provider_data_api_key_field: str = "vllm_api_token"

    def get_api_key(self) -> str | None:
        if self.config.auth_credential:
            return self.config.auth_credential.get_secret_value()
        return "NO KEY REQUIRED"

    def get_base_url(self) -> str:
        """Get the base URL from config."""
        if not self.config.base_url:
            raise ValueError("No base URL configured")
        return str(self.config.base_url)

    async def initialize(self) -> None:
        if not self.config.base_url:
            raise ValueError(
                "You must provide a URL in config.yaml (or via the VLLM_URL environment variable) to use vLLM."
            )

    async def health(self) -> HealthResponse:
        """
        Performs a health check by verifying connectivity to the remote vLLM server.
        This method is used by the Provider API to verify
        that the service is running correctly.
        Uses the unauthenticated /health endpoint.
        Returns:

            HealthResponse: A dictionary containing the health status.
        """
        try:
            base_url = self.get_base_url()
            health_url = urljoin(base_url, "health")

            async with httpx.AsyncClient() as client:
                response = await client.get(health_url)
                response.raise_for_status()
                return HealthResponse(status=HealthStatus.OK)
        except Exception as e:
            return HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}")

    async def check_model_availability(self, model: str) -> bool:
        """
        Skip the check when running without authentication.
        """
        if not self.config.auth_credential:
            model_ids = []
            async for m in self.client.models.list():
                if m.id == model:  # Found exact match
                    return True
                model_ids.append(m.id)
            raise ValueError(f"Model '{model}' not found. Available models: {model_ids}")
        log.warning(f"Not checking model availability for {model} as API token may trigger OAuth workflow")
        return True

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        params = params.model_copy()

        # Apply vLLM-specific defaults
        if params.max_tokens is None and self.config.max_tokens:
            params.max_tokens = self.config.max_tokens

        # This is to be consistent with OpenAI API and support vLLM <= v0.6.3
        # References:
        #   * https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
        #   * https://github.com/vllm-project/vllm/pull/10000
        if not params.tools and params.tool_choice is not None:
            params.tool_choice = ToolChoice.none.value

        return await super().openai_chat_completion(params)

    def construct_model_from_identifier(self, identifier: str) -> Model:
        if (
            "rerank" in identifier.lower()
        ):  # TODO: guessing that any model with "rerank" in its name is a reranking model is pretty hacky
            return Model(
                provider_id=self.__provider_id__,  # type: ignore[attr-defined]
                provider_resource_id=identifier,
                identifier=identifier,
                model_type=ModelType.rerank,
            )
        return super().construct_model_from_identifier(identifier)

    async def rerank(
        self,
        request: RerankRequest,
    ) -> RerankResponse:
        def format_item(
            item: str | OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam,
        ) -> str:
            if isinstance(item, str):
                return item
            elif isinstance(item, OpenAIChatCompletionContentPartTextParam):
                return item.text
            elif isinstance(item, OpenAIChatCompletionContentPartImageParam):
                raise ValueError("vLLM rerank API does not support images")
            else:
                raise ValueError("Unsupported item type for reranking")

        payload: dict[str, str | int | float | list[str]] = {
            "model": request.model,
            "query": format_item(request.query),
            "documents": [format_item(item) for item in request.items],
        }
        if request.max_num_results is not None:
            payload["top_n"] = request.max_num_results

        try:
            async with aiohttp.ClientSession() as session:
                # vLLM does not support /v1/rerank ->
                #   "To indicate that the rerank API is not part of the standard OpenAI API,
                #    we have located it at `/rerank`. Please update your client accordingly.
                #    (Note: Conforms to JinaAI rerank API)" - vLLM 0.15.1
                endpoint = self.get_base_url().replace("/v1", "") + "/rerank"  # TODO: find a better solution
                async with session.post(endpoint, headers={}, json=payload) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        raise RuntimeError(
                            f"vLLM rerank API request failed with status {response.status}: {response_text}"
                        )

                    def convert_result_item(item: dict) -> RerankData:
                        if "index" not in item or "relevance_score" not in item:
                            raise RuntimeError(
                                "vLLM rerank API response missing required fields 'index' or 'relevance_score'"
                            )

                        try:
                            return RerankData(index=int(item["index"]), relevance_score=float(item["relevance_score"]))
                        except (TypeError, ValueError) as e:
                            raise RuntimeError(f"Invalid data types in vLLM rerank API response: {e}") from e

                    result = await response.json()

                    if "results" not in result:
                        raise RuntimeError("vLLM rerank API response missing 'results' field")

                    rerank_data = [convert_result_item(item) for item in result.get("results")]
                    rerank_data.sort(key=lambda entry: entry.relevance_score, reverse=True)

                    return RerankResponse(data=rerank_data)

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to vLLM rerank API at {endpoint}: {e}") from e
