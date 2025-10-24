# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from collections.abc import Iterable

from huggingface_hub import AsyncInferenceClient, HfApi
from pydantic import SecretStr

from llama_stack.apis.inference import (
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import InferenceAPIImplConfig, InferenceEndpointImplConfig, TGIImplConfig

log = get_logger(name=__name__, category="inference::tgi")


class _HfAdapter(OpenAIMixin):
    url: str
    api_key: SecretStr

    hf_client: AsyncInferenceClient
    max_tokens: int
    model_id: str

    overwrite_completion_id = True  # TGI always returns id=""

    def get_api_key(self):
        return "NO KEY REQUIRED"

    def get_base_url(self):
        return self.url

    async def list_provider_model_ids(self) -> Iterable[str]:
        return [self.model_id]

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError()


class TGIAdapter(_HfAdapter):
    async def initialize(self, config: TGIImplConfig) -> None:
        if not config.url:
            raise ValueError("You must provide a URL in run.yaml (or via the TGI_URL environment variable) to use TGI.")
        log.info(f"Initializing TGI client with url={config.url}")
        self.hf_client = AsyncInferenceClient(model=config.url, provider="hf-inference")
        endpoint_info = await self.hf_client.get_endpoint_info()
        self.max_tokens = endpoint_info["max_total_tokens"]
        self.model_id = endpoint_info["model_id"]
        self.url = f"{config.url.rstrip('/')}/v1"
        self.api_key = SecretStr("NO_KEY")


class InferenceAPIAdapter(_HfAdapter):
    async def initialize(self, config: InferenceAPIImplConfig) -> None:
        self.hf_client = AsyncInferenceClient(model=config.huggingface_repo, token=config.api_token.get_secret_value())
        endpoint_info = await self.hf_client.get_endpoint_info()
        self.max_tokens = endpoint_info["max_total_tokens"]
        self.model_id = endpoint_info["model_id"]
        # TODO: how do we set url for this?


class InferenceEndpointAdapter(_HfAdapter):
    async def initialize(self, config: InferenceEndpointImplConfig) -> None:
        # Get the inference endpoint details
        api = HfApi(token=config.api_token.get_secret_value())
        endpoint = api.get_inference_endpoint(config.endpoint_name)
        # Wait for the endpoint to be ready (if not already)
        endpoint.wait(timeout=60)

        # Initialize the adapter
        self.hf_client = endpoint.async_client
        self.model_id = endpoint.repository
        self.max_tokens = int(endpoint.raw["model"]["image"]["custom"]["env"]["MAX_TOTAL_TOKENS"])
        # TODO: how do we set url for this?
