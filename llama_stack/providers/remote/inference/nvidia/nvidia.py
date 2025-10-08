# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from openai import NOT_GIVEN

from llama_stack.apis.inference import (
    OpenAIEmbeddingData,
    OpenAIEmbeddingsResponse,
    OpenAIEmbeddingUsage,
)
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

    async def initialize(self) -> None:
        logger.info(f"Initializing NVIDIAInferenceAdapter({self.config.url})...")

        if _is_nvidia_hosted(self.config):
            if not self.config.api_key:
                raise RuntimeError(
                    "API key is required for hosted NVIDIA NIM. Either provide an API key or use a self-hosted NIM."
                )

    def get_api_key(self) -> str:
        """
        Get the API key for OpenAI mixin.

        :return: The NVIDIA API key
        """
        return self.config.api_key.get_secret_value() if self.config.api_key else "NO KEY"

    def get_base_url(self) -> str:
        """
        Get the base URL for OpenAI mixin.

        :return: The NVIDIA API base URL
        """
        return f"{self.config.url}/v1" if self.config.append_api_version else self.config.url

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        """
        OpenAI-compatible embeddings for NVIDIA NIM.

        Note: NVIDIA NIM asymmetric embedding models require an "input_type" field not present in the standard OpenAI embeddings API.
        We default this to "query" to ensure requests succeed when using the
        OpenAI-compatible endpoint. For passage embeddings, use the embeddings API with
        `task_type='document'`.
        """
        extra_body: dict[str, object] = {"input_type": "query"}
        logger.warning(
            "NVIDIA OpenAI-compatible embeddings: defaulting to input_type='query'. "
            "For passage embeddings, use the embeddings API with task_type='document'."
        )

        response = await self.client.embeddings.create(
            model=await self._get_provider_model_id(model),
            input=input,
            encoding_format=encoding_format if encoding_format is not None else NOT_GIVEN,
            dimensions=dimensions if dimensions is not None else NOT_GIVEN,
            user=user if user is not None else NOT_GIVEN,
            extra_body=extra_body,
        )

        data = []
        for i, embedding_data in enumerate(response.data):
            data.append(
                OpenAIEmbeddingData(
                    embedding=embedding_data.embedding,
                    index=i,
                )
            )

        usage = OpenAIEmbeddingUsage(
            prompt_tokens=response.usage.prompt_tokens,
            total_tokens=response.usage.total_tokens,
        )

        return OpenAIEmbeddingsResponse(
            data=data,
            model=response.model,
            usage=usage,
        )
