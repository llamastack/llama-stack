# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import oci
from oci.generative_ai.generative_ai_client import GenerativeAiClient
from oci.generative_ai.models import ModelCollection

from llama_stack.apis.models import ModelType
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper, ProviderModelEntry


def build_oci_model_entries(
    compartment_id: str,
    oci_config: dict | None = None,
    oci_signer: oci.auth.signers.InstancePrincipalsSecurityTokenSigner | None = None,
) -> list[ProviderModelEntry]:
    """
    Build OCI model entries by fetching available models from OCI Generative AI service.

    Args:
        compartment_id: OCI compartment ID where models are located
        config: OCI config dict (if None, will use config file)
        config_profile: OCI config profile to use (defaults to "CHICAGO" or OCI_CLI_PROFILE env var)

    Returns:
        List of ProviderModelEntry objects mapping display_name to model.id
    """
    if oci_signer is None:
        client = GenerativeAiClient(config=oci_config)
    else:
        client = GenerativeAiClient(config=oci_config, signer=oci_signer)

    print("Here 1")
    models: ModelCollection = client.list_models(compartment_id=compartment_id, capability=["CHAT"]).data

    model_entries = []
    seen_models = set()
    for model in models.items:
        if model.time_deprecated or model.time_on_demand_retired:
            continue

        if "CHAT" not in model.capabilities or "FINE_TUNE" in model.capabilities:
            continue

        if model.display_name in seen_models:
            continue

        seen_models.add(model.display_name)

        entry = ProviderModelEntry(
            provider_model_id=model.id,
            aliases=[model.display_name],
            model_type=ModelType.llm,
            metadata={
                "display_name": model.display_name,
                "capabilities": model.capabilities,
                "oci_model_id": model.id,
            },
        )

        seen_models.add(model.display_name)

        model_entries.append(entry)

    return model_entries


class OCIModelRegistryHelper(ModelRegistryHelper):
    """
    OCI-specific model registry helper that dynamically fetches models from OCI.
    """

    def __init__(
        self,
        compartment_id: str,
        oci_config: dict | None = None,
        oci_signer: oci.auth.signers.InstancePrincipalsSecurityTokenSigner | None = None,
        allowed_models: list[str] | None = None,
    ):
        model_entries = build_oci_model_entries(compartment_id, oci_config, oci_signer)

        super().__init__(model_entries=model_entries, allowed_models=allowed_models)

        self.compartment_id = compartment_id
        self.oci_config = oci_config
        self.oci_signer = oci_signer

    async def should_refresh_models(self) -> bool:
        return True

    async def check_model_availability(self, alias: str) -> bool:
        client = GenerativeAiClient(config=self.oci_config, signer=self.oci_signer)
        model_id = self.get_provider_model_id(alias)
        response = client.get_model(model_id)
        return response.data is not None


# For backward compatibility, create an empty MODEL_ENTRIES list
# The actual models will be built dynamically by OCIModelRegistryHelper
MODEL_ENTRIES: list[ProviderModelEntry] = []
