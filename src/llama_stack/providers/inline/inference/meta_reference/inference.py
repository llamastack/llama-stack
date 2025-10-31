# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from collections.abc import AsyncIterator

from llama_stack.apis.inference import (
    InferenceProvider,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletionRequestWithExtraBody,
)
from llama_stack.apis.inference.inference import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAICompletion,
)
from llama_stack.apis.models import Model, ModelType
from llama_stack.log import get_logger
from llama_stack.models.llama.llama3.chat_format import ChatFormat as Llama3ChatFormat
from llama_stack.models.llama.llama3.tokenizer import Tokenizer as Llama3Tokenizer
from llama_stack.models.llama.llama4.chat_format import ChatFormat as Llama4ChatFormat
from llama_stack.models.llama.llama4.tokenizer import Tokenizer as Llama4Tokenizer
from llama_stack.models.llama.sku_list import resolve_model
from llama_stack.models.llama.sku_types import ModelFamily
from llama_stack.providers.datatypes import ModelsProtocolPrivate
from llama_stack.providers.utils.inference.embedding_mixin import (
    SentenceTransformerEmbeddingMixin,
)
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
    build_hf_repo_model_entry,
)

from .config import MetaReferenceInferenceConfig
from .generators import LlamaGenerator
from .model_parallel import LlamaModelParallelGenerator

log = get_logger(__name__, category="inference")
# there's a single model parallel process running serving the model. for now,
# we don't support multiple concurrent requests to this process.
SEMAPHORE = asyncio.Semaphore(1)


def llama_builder_fn(config: MetaReferenceInferenceConfig, model_id: str, llama_model: Model) -> LlamaGenerator:
    return LlamaGenerator(config, model_id, llama_model)


class MetaReferenceInferenceImpl(
    SentenceTransformerEmbeddingMixin,
    InferenceProvider,
    ModelsProtocolPrivate,
):
    def __init__(self, config: MetaReferenceInferenceConfig) -> None:
        self.config = config
        self.model_id = None
        self.llama_model = None

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        if self.config.create_distributed_process_group:
            self.generator.stop()

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion:
        raise NotImplementedError("OpenAI completion not supported by meta reference provider")

    async def should_refresh_models(self) -> bool:
        return False

    async def list_models(self) -> list[Model] | None:
        return None

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def register_model(self, model: Model) -> Model:
        llama_model = (
            resolve_model(model.metadata["llama_model"])
            if "llama_model" in model.metadata
            else resolve_model(model.identifier)
        )
        if llama_model is None:
            raise ValueError(
                "Please make sure your llama_model in model metadata or model identifier is in Llama SKU list"
            )

        self.model_registry_helper = ModelRegistryHelper(
            [
                build_hf_repo_model_entry(
                    llama_model.descriptor(),
                    llama_model.core_model_id.value,
                )
            ],
        )
        model = await self.model_registry_helper.register_model(model)

        if model.model_type == ModelType.embedding:
            self._load_sentence_transformer_model(model.provider_resource_id)

        # TODO: what is this?! you can't really specify skipping via model metadata
        # kill this madness
        if "skip_load" in model.metadata and model.metadata["skip_load"]:
            return model

        await self.load_model(model.identifier, llama_model)
        return model

    async def load_model(self, model_id, llama_model) -> None:
        log.info(f"Loading model `{model_id}`")

        builder_params = [self.config, model_id, llama_model]

        if self.config.create_distributed_process_group:
            self.generator = LlamaModelParallelGenerator(
                model_parallel_size=self.config.model_parallel_size or llama_model.pth_file_count,
                builder_fn=llama_builder_fn,
                builder_params=builder_params,
                formatter=(
                    Llama4ChatFormat(Llama4Tokenizer.get_instance())
                    if llama_model.model_family == ModelFamily.llama4
                    else Llama3ChatFormat(Llama3Tokenizer.get_instance())
                ),
            )
            self.generator.start()
        else:
            self.generator = llama_builder_fn(*builder_params)

        self.model_id = model_id
        self.llama_model = llama_model

        log.info("Warming up...")
        await self.openai_chat_completion(
            model=model_id,
            messages=[{"role": "user", "content": "Hi how are you?"}],
            max_tokens=20,
        )
        log.info("Warmed up!")

    def check_model(self, request) -> None:
        if self.model_id is None or self.llama_model is None:
            raise RuntimeError(
                "No avaible model yet, please register your requested model or add your model in the resouces first"
            )
        elif request.model != self.model_id:
            raise RuntimeError(f"Model mismatch: request model: {request.model} != loaded model: {self.model_id}")

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        raise NotImplementedError("OpenAI chat completion not supported by meta-reference inference provider")
