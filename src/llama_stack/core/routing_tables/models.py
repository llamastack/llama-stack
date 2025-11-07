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
)
from llama_stack.core.request_headers import PROVIDER_DATA_VAR, NeedsRequestProviderData
from llama_stack.core.utils.dynamic import instantiate_class_type
from llama_stack.log import get_logger

from .common import CommonRoutingTableImpl, lookup_model

logger = get_logger(name=__name__, category="core::routing_tables")


class ModelsRoutingTable(CommonRoutingTableImpl, Models):
    listed_providers: set[str] = set()

    async def refresh(self) -> None:
        for provider_id, provider in self.impls_by_provider_id.items():
            refresh = await provider.should_refresh_models()
            refresh = refresh or provider_id not in self.listed_providers
            if not refresh:
                continue

            try:
                models = await provider.list_models()
            except Exception as e:
                logger.warning(f"Model refresh failed for provider {provider_id}: {e}")
                continue

            self.listed_providers.add(provider_id)
            if models is None:
                continue

            await self.update_registered_models(provider_id, models)

    async def _get_dynamic_models_from_provider_data(self) -> list[Model]:
        """
        Fetch models from providers that have credentials in the current request's provider_data.

        This allows users to see models available to them from providers that require
        per-request API keys (via X-LlamaStack-Provider-Data header).

        Returns models with fully qualified identifiers (provider_id/model_id) but does NOT
        cache them in the registry since they are user-specific.
        """
        provider_data = PROVIDER_DATA_VAR.get()
        if not provider_data:
            return []

        dynamic_models = []

        for provider_id, provider in self.impls_by_provider_id.items():
            # Check if this provider supports provider_data
            if not isinstance(provider, NeedsRequestProviderData):
                continue

            # Check if provider has a validator (some providers like ollama don't need per-request credentials)
            spec = getattr(provider, "__provider_spec__", None)
            if not spec or not getattr(spec, "provider_data_validator", None):
                continue

            # Validate provider_data silently - we're speculatively checking all providers
            # so validation failures are expected when user didn't provide keys for this provider
            try:
                validator = instantiate_class_type(spec.provider_data_validator)
                validator(**provider_data)
            except Exception:
                # User didn't provide credentials for this provider - skip silently
                continue

            # Validation succeeded! User has credentials for this provider
            # Now try to list models
            try:
                models = await provider.list_models()
                if not models:
                    continue

                # Ensure models have fully qualified identifiers with provider_id prefix
                for model in models:
                    # Only add prefix if model identifier doesn't already have it
                    if not model.identifier.startswith(f"{provider_id}/"):
                        model.identifier = f"{provider_id}/{model.provider_resource_id}"

                    dynamic_models.append(model)

                logger.debug(f"Fetched {len(models)} models from provider {provider_id} using provider_data")

            except Exception as e:
                logger.debug(f"Failed to list models from provider {provider_id} with provider_data: {e}")
                continue

        return dynamic_models

    async def list_models(self) -> ListModelsResponse:
        # Get models from registry
        registry_models = await self.get_all_with_type("model")

        # Get additional models available via provider_data (user-specific, not cached)
        dynamic_models = await self._get_dynamic_models_from_provider_data()

        # Combine, avoiding duplicates (registry takes precedence)
        registry_identifiers = {m.identifier for m in registry_models}
        unique_dynamic_models = [m for m in dynamic_models if m.identifier not in registry_identifiers]

        return ListModelsResponse(data=registry_models + unique_dynamic_models)

    async def openai_list_models(self) -> OpenAIListModelsResponse:
        # Get models from registry
        registry_models = await self.get_all_with_type("model")

        # Get additional models available via provider_data (user-specific, not cached)
        dynamic_models = await self._get_dynamic_models_from_provider_data()

        # Combine, avoiding duplicates (registry takes precedence)
        registry_identifiers = {m.identifier for m in registry_models}
        unique_dynamic_models = [m for m in dynamic_models if m.identifier not in registry_identifiers]

        all_models = registry_models + unique_dynamic_models

        openai_models = [
            OpenAIModel(
                id=model.identifier,
                object="model",
                created=int(time.time()),
                owned_by="llama_stack",
                custom_metadata={
                    "model_type": model.model_type,
                    "provider_id": model.provider_id,
                    "provider_resource_id": model.provider_resource_id,
                    **model.metadata,
                },
            )
            for model in all_models
        ]
        return OpenAIListModelsResponse(data=openai_models)

    async def get_model(self, model_id: str) -> Model:
        return await lookup_model(self, model_id)

    async def get_provider_impl(self, model_id: str) -> Any:
        model = await lookup_model(self, model_id)
        if model.provider_id not in self.impls_by_provider_id:
            raise ValueError(f"Provider {model.provider_id} not found in the routing table")
        return self.impls_by_provider_id[model.provider_id]

    async def has_model(self, model_id: str) -> bool:
        """
        Check if a model exists in the routing table.

        :param model_id: The model identifier to check
        :return: True if the model exists, False otherwise
        """
        try:
            await lookup_model(self, model_id)
            return True
        except ModelNotFoundError:
            return False

    async def register_model(
        self,
        model_id: str,
        provider_model_id: str | None = None,
        provider_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        model_type: ModelType | None = None,
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

        identifier = f"{provider_id}/{provider_model_id}"
        model = ModelWithOwner(
            identifier=identifier,
            provider_resource_id=provider_model_id,
            provider_id=provider_id,
            metadata=metadata,
            model_type=model_type,
            source=RegistryEntrySource.via_register_api,
        )
        registered_model = await self.register_object(model)
        return registered_model

    async def unregister_model(self, model_id: str) -> None:
        existing_model = await self.get_model(model_id)
        if existing_model is None:
            raise ModelNotFoundError(model_id)
        await self.unregister_object(existing_model)

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
