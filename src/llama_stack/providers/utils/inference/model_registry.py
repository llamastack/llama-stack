# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, SecretStr

from llama_stack.apis.common.errors import UnsupportedModelError
from llama_stack.apis.models import ModelType
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import Model, ModelsProtocolPrivate
from llama_stack.providers.utils.inference import (
    ALL_HUGGINGFACE_REPOS_TO_MODEL_DESCRIPTOR,
)

logger = get_logger(name=__name__, category="providers::utils")


class RemoteInferenceProviderConfig(BaseModel):
    allowed_models: list[str] | None = Field(
        default=None,
        description="List of models that should be registered with the model registry. If None, all models are allowed.",
    )
    refresh_models: bool = Field(
        default=False,
        description="Whether to refresh models periodically from the provider",
    )
    auth_credential: SecretStr | None = Field(
        default=None,
        description="Authentication credential for the provider",
        alias="api_key",
    )


# TODO: this class is more confusing than useful right now. We need to make it
# more closer to the Model class.
class ProviderModelEntry(BaseModel):
    provider_model_id: str
    aliases: list[str] = Field(default_factory=list)
    llama_model: str | None = None
    model_type: ModelType = ModelType.llm
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_hf_repo_model_entry(
    provider_model_id: str,
    model_descriptor: str,
    additional_aliases: list[str] | None = None,
) -> ProviderModelEntry:
    aliases = [
        # NOTE: avoid HF aliases because they _cannot_ be unique across providers
        # get_huggingface_repo(model_descriptor),
    ]
    if additional_aliases:
        aliases.extend(additional_aliases)
    aliases = [alias for alias in aliases if alias is not None]
    return ProviderModelEntry(
        provider_model_id=provider_model_id,
        aliases=aliases,
        llama_model=model_descriptor,
    )


class ModelRegistryHelper(ModelsProtocolPrivate):
    __provider_id__: str

    def __init__(
        self,
        model_entries: list[ProviderModelEntry] | None = None,
        allowed_models: list[str] | None = None,
    ):
        self.allowed_models = allowed_models if allowed_models else []

        self.alias_to_provider_id_map = {}
        self.provider_id_to_llama_model_map = {}
        self.model_entries = model_entries or []
        for entry in self.model_entries:
            for alias in entry.aliases:
                self.alias_to_provider_id_map[alias] = entry.provider_model_id

            # also add a mapping from provider model id to itself for easy lookup
            self.alias_to_provider_id_map[entry.provider_model_id] = entry.provider_model_id

            if entry.llama_model:
                self.alias_to_provider_id_map[entry.llama_model] = entry.provider_model_id
                self.provider_id_to_llama_model_map[entry.provider_model_id] = entry.llama_model

    async def list_models(self) -> list[Model] | None:
        models = []
        for entry in self.model_entries:
            ids = [entry.provider_model_id] + entry.aliases
            for id in ids:
                if self.allowed_models and id not in self.allowed_models:
                    continue
                models.append(
                    Model(
                        identifier=id,
                        provider_resource_id=entry.provider_model_id,
                        model_type=entry.model_type,
                        metadata=entry.metadata,
                        provider_id=self.__provider_id__,
                    )
                )
        return models

    async def should_refresh_models(self) -> bool:
        return False

    def get_provider_model_id(self, identifier: str) -> str | None:
        return self.alias_to_provider_id_map.get(identifier, None)

    # TODO: why keep a separate llama model mapping?
    def get_llama_model(self, provider_model_id: str) -> str | None:
        return self.provider_id_to_llama_model_map.get(provider_model_id, None)

    async def check_model_availability(self, model: str) -> bool:
        """
        Check if a specific model is available from the provider (non-static check).

        This is for subclassing purposes, so providers can check if a specific
        model is currently available for use through dynamic means (e.g., API calls).

        This method should NOT check statically configured model entries in
        `self.alias_to_provider_id_map` - that is handled separately in register_model.

        Default implementation returns False (no dynamic models available).

        :param model: The model identifier to check.
        :return: True if the model is available dynamically, False otherwise.
        """
        logger.info(
            f"check_model_availability is not implemented for {self.__class__.__name__}. Returning False by default."
        )
        return False

    async def register_model(self, model: Model) -> Model:
        # Check if model is supported in static configuration
        supported_model_id = self.get_provider_model_id(model.provider_resource_id)

        # If not found in static config, check if it's available dynamically from provider
        if not supported_model_id:
            if await self.check_model_availability(model.provider_resource_id):
                supported_model_id = model.provider_resource_id
            else:
                # note: we cannot provide a complete list of supported models without
                #       getting a complete list from the provider, so we return "..."
                all_supported_models = [*self.alias_to_provider_id_map.keys(), "..."]
                raise UnsupportedModelError(model.provider_resource_id, all_supported_models)

        provider_resource_id = self.get_provider_model_id(model.model_id)
        if model.model_type == ModelType.embedding:
            # embedding models are always registered by their provider model id and does not need to be mapped to a llama model
            provider_resource_id = model.provider_resource_id
        if provider_resource_id:
            if provider_resource_id != supported_model_id:  # be idempotent, only reject differences
                raise ValueError(
                    f"Model id '{model.model_id}' is already registered. Please use a different id or unregister it first."
                )
        else:
            llama_model = model.metadata.get("llama_model")
            if llama_model:
                existing_llama_model = self.get_llama_model(model.provider_resource_id)
                if existing_llama_model:
                    if existing_llama_model != llama_model:
                        raise ValueError(
                            f"Provider model id '{model.provider_resource_id}' is already registered to a different llama model: '{existing_llama_model}'"
                        )
                else:
                    if llama_model not in ALL_HUGGINGFACE_REPOS_TO_MODEL_DESCRIPTOR:
                        raise ValueError(
                            f"Invalid llama_model '{llama_model}' specified in metadata. "
                            f"Must be one of: {', '.join(ALL_HUGGINGFACE_REPOS_TO_MODEL_DESCRIPTOR.keys())}"
                        )
                    self.provider_id_to_llama_model_map[model.provider_resource_id] = (
                        ALL_HUGGINGFACE_REPOS_TO_MODEL_DESCRIPTOR[llama_model]
                    )

        # Register the model alias, ensuring it maps to the correct provider model id
        self.alias_to_provider_id_map[model.model_id] = supported_model_id

        return model

    async def unregister_model(self, model_id: str) -> None:
        # model_id is the identifier, not the provider_resource_id
        # unfortunately, this ID can be of the form provider_id/model_id which
        # we never registered. TODO: fix this by significantly rewriting
        # registration and registry helper
        pass
