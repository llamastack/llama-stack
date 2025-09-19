# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
from typing import Any

from llama_stack.apis.common.errors import ModelNotFoundError
from llama_stack.apis.models import ListModelsResponse, Model, Models, ModelType, OpenAIListModelsResponse, OpenAIModel
from llama_stack.core.datatypes import (
    ModelWithOwner,
    RegistryEntrySource,
    StackRunConfig,
)
from llama_stack.log import get_logger

from .common import CommonRoutingTableImpl, lookup_model

logger = get_logger(name=__name__, category="core::routing_tables")


class ModelsRoutingTable(CommonRoutingTableImpl, Models):
    listed_providers: set[str] = set()
    current_run_config: "StackRunConfig | None" = None

    async def refresh(self) -> None:
        for provider_id, provider in self.impls_by_provider_id.items():
            refresh = await provider.should_refresh_models()
            refresh = refresh or provider_id not in self.listed_providers
            if not refresh:
                continue

            try:
                models = await provider.list_models()
            except Exception as e:
                logger.exception(f"Model refresh failed for provider {provider_id}: {e}")
                continue

            self.listed_providers.add(provider_id)
            if models is None:
                continue

            await self.update_registered_models(provider_id, models)

    async def list_models(self) -> ListModelsResponse:
        return ListModelsResponse(data=await self.get_all_with_type("model"))

    async def openai_list_models(self) -> OpenAIListModelsResponse:
        models = await self.get_all_with_type("model")
        openai_models = [
            OpenAIModel(
                id=model.identifier,
                object="model",
                created=int(time.time()),
                owned_by="llama_stack",
            )
            for model in models
        ]
        return OpenAIListModelsResponse(data=openai_models)

    async def get_model(self, model_id: str) -> Model:
        return await lookup_model(self, model_id)

    async def get_provider_impl(self, model_id: str) -> Any:
        model = await lookup_model(self, model_id)
        if model.provider_id not in self.impls_by_provider_id:
            raise ValueError(f"Provider {model.provider_id} not found in the routing table")
        return self.impls_by_provider_id[model.provider_id]

    async def register_model(
        self,
        model_id: str,
        provider_model_id: str | None = None,
        provider_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        model_type: ModelType | None = None,
        source: RegistryEntrySource = RegistryEntrySource.via_register_api,
    ) -> Model:
        if provider_id is None:
            # If provider_id not specified, use the only provider if it supports this model
            if len(self.impls_by_provider_id) == 1:
                provider_id = list(self.impls_by_provider_id.keys())[0]
            else:
                raise ValueError(
                    f"Please specify a provider_id for model {model_id} since multiple providers are available: {self.impls_by_provider_id.keys()}.\n\n"
                    "Use the provider_id as a prefix to disambiguate, e.g. 'provider_id/model_id'."
                )

        provider_model_id = provider_model_id or model_id
        metadata = metadata or {}
        model_type = model_type or ModelType.llm
        if "embedding_dimension" not in metadata and model_type == ModelType.embedding:
            raise ValueError("Embedding model must have an embedding dimension in its metadata")

        # an identifier different than provider_model_id implies it is an alias, so that
        # becomes the globally unique identifier. otherwise provider_model_ids can conflict,
        # so as a general rule we must use the provider_id to disambiguate.

        if model_id != provider_model_id:
            identifier = model_id
        else:
            identifier = f"{provider_id}/{provider_model_id}"

        model = ModelWithOwner(
            identifier=identifier,
            provider_resource_id=provider_model_id,
            provider_id=provider_id,
            metadata=metadata,
            model_type=model_type,
            source=source,
        )
        registered_model = await self.register_object(model)
        return registered_model

    async def unregister_model(self, model_id: str) -> None:
        existing_model = await self.get_model(model_id)
        if existing_model is None:
            raise ModelNotFoundError(model_id)
        await self.unregister_object(existing_model)

    async def cleanup_disabled_provider_models(self) -> None:
        """Remove models from providers that are no longer enabled in the current run config."""
        if not self.current_run_config:
            return

        # Get enabled provider IDs from the current run config
        enabled_provider_ids = set()
        for _api, providers in self.current_run_config.providers.items():
            for provider in providers:
                if provider.provider_id and provider.provider_id != "__disabled__":
                    enabled_provider_ids.add(provider.provider_id)

        # Get all existing models
        existing_models = await self.get_all_with_type("model")

        # Find models from disabled providers (excluding user-registered models)
        models_to_remove = []
        for model in existing_models:
            if model.provider_id not in enabled_provider_ids and model.source != RegistryEntrySource.via_register_api:
                models_to_remove.append(model)

        # Remove the models
        for model in models_to_remove:
            logger.info(f"Removing model {model.identifier} from disabled provider {model.provider_id}")
            await self.unregister_object(model)

    async def register_from_config_models(self) -> None:
        """Register from_config models from the current run configuration."""
        if not self.current_run_config:
            return

        # Register new from_config models (old ones automatically disappear since they're not persisted)
        for model_input in self.current_run_config.models:
            # Skip models with disabled providers
            if not model_input.provider_id or model_input.provider_id == "__disabled__":
                continue

            # Generate identifier
            if model_input.model_id != (model_input.provider_model_id or model_input.model_id):
                identifier = model_input.model_id
            else:
                identifier = f"{model_input.provider_id}/{model_input.provider_model_id or model_input.model_id}"

            model = ModelWithOwner(
                identifier=identifier,
                provider_resource_id=model_input.provider_model_id or model_input.model_id,
                provider_id=model_input.provider_id,
                metadata=model_input.metadata,
                model_type=model_input.model_type or ModelType.llm,
                source=RegistryEntrySource.from_config,
            )

            # Register the model (will be cached in memory but not persisted to disk)
            await self.dist_registry.register(model)

    async def update_registered_models(
        self,
        provider_id: str,
        models: list[Model],
    ) -> None:
        existing_models = await self.get_all_with_type("model")

        # we may have an alias for the model registered by the user (or during initialization
        # from run.yaml) that we need to keep track of
        model_ids = {}
        for model in existing_models:
            if model.provider_id != provider_id:
                continue
            if model.source == RegistryEntrySource.via_register_api:
                model_ids[model.provider_resource_id] = model.identifier
                continue

            logger.debug(f"unregistering model {model.identifier}")
            await self.unregister_object(model)

        for model in models:
            if model.provider_resource_id in model_ids:
                # avoid overwriting a non-provider-registered model entry
                continue

            if model.identifier == model.provider_resource_id:
                model.identifier = f"{provider_id}/{model.provider_resource_id}"

            logger.debug(f"registering model {model.identifier} ({model.provider_resource_id})")
            await self.register_object(
                ModelWithOwner(
                    identifier=model.identifier,
                    provider_resource_id=model.provider_resource_id,
                    provider_id=provider_id,
                    metadata=model.metadata,
                    model_type=model.model_type,
                    source=RegistryEntrySource.listed_from_provider,
                )
            )
