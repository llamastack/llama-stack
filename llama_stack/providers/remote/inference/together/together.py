# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from collections.abc import Iterable

from together import AsyncTogether
from together.constants import BASE_URL

from llama_stack.apis.inference import (
    OpenAIEmbeddingsResponse,
)
from llama_stack.apis.inference.inference import OpenAIEmbeddingUsage
from llama_stack.apis.models import Model
from llama_stack.core.request_headers import NeedsRequestProviderData
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import TogetherImplConfig

logger = get_logger(name=__name__, category="inference::together")


class TogetherInferenceAdapter(OpenAIMixin, NeedsRequestProviderData):
    config: TogetherImplConfig

    embedding_model_metadata: dict[str, dict[str, int]] = {
        "togethercomputer/m2-bert-80M-32k-retrieval": {"embedding_dimension": 768, "context_length": 32768},
        "BAAI/bge-large-en-v1.5": {"embedding_dimension": 1024, "context_length": 512},
        "BAAI/bge-base-en-v1.5": {"embedding_dimension": 768, "context_length": 512},
        "Alibaba-NLP/gte-modernbert-base": {"embedding_dimension": 768, "context_length": 8192},
        "intfloat/multilingual-e5-large-instruct": {"embedding_dimension": 1024, "context_length": 512},
    }

    _model_cache: dict[str, Model] = {}

    provider_data_api_key_field: str = "together_api_key"

    def get_api_key(self):
        return self.config.api_key.get_secret_value() if self.config.api_key else None

    def get_base_url(self):
        return BASE_URL

    def _get_client(self) -> AsyncTogether:
        together_api_key = None
        config_api_key = self.config.api_key.get_secret_value() if self.config.api_key else None
        if config_api_key:
            together_api_key = config_api_key
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.together_api_key:
                raise ValueError(
                    'Pass Together API Key in the header X-LlamaStack-Provider-Data as { "together_api_key": <your api key>}'
                )
            together_api_key = provider_data.together_api_key
        return AsyncTogether(api_key=together_api_key)

    async def list_provider_model_ids(self) -> Iterable[str]:
        # Together's /v1/models is not compatible with OpenAI's /v1/models. Together support ticket #13355 -> will not fix, use Together's own client
        return [m.id for m in await self._get_client().models.list()]

    async def should_refresh_models(self) -> bool:
        return True

    async def check_model_availability(self, model):
        return model in self._model_cache

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        """
        Together's OpenAI-compatible embeddings endpoint is not compatible with
        the standard OpenAI embeddings endpoint.

        The endpoint -
         - not all models return usage information
         - does not support user param, returns 400 Unrecognized request arguments supplied: user
         - does not support dimensions param, returns 400 Unrecognized request arguments supplied: dimensions
        """
        # Together support ticket #13332 -> will not fix
        if user is not None:
            raise ValueError("Together's embeddings endpoint does not support user param.")
        # Together support ticket #13333 -> escalated
        if dimensions is not None:
            raise ValueError("Together's embeddings endpoint does not support dimensions param.")

        response = await self.client.embeddings.create(
            model=await self._get_provider_model_id(model),
            input=input,
            encoding_format=encoding_format,
        )

        response.model = model  # return the user the same model id they provided, avoid exposing the provider model id

        # Together support ticket #13330 -> escalated
        #  - togethercomputer/m2-bert-80M-32k-retrieval *does not* return usage information
        if not hasattr(response, "usage") or response.usage is None:
            logger.warning(
                f"Together's embedding endpoint for {model} did not return usage information, substituting -1s."
            )
            response.usage = OpenAIEmbeddingUsage(prompt_tokens=-1, total_tokens=-1)

        return response  # type: ignore[no-any-return]
