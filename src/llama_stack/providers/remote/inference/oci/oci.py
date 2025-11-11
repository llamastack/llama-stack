# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from collections.abc import Iterable
from typing import Any

import httpx
import oci
from oci.generative_ai.generative_ai_client import GenerativeAiClient
from oci.generative_ai.models import ModelCollection
from openai._base_client import DefaultAsyncHttpxClient

from llama_stack.apis.inference.inference import (
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)
from llama_stack.apis.models import ModelType
from llama_stack.log import get_logger
from llama_stack.providers.remote.inference.oci.auth import OciInstancePrincipalAuth, OciUserPrincipalAuth
from llama_stack.providers.remote.inference.oci.config import OCIConfig
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin

logger = get_logger(name=__name__, category="inference::oci")

OCI_AUTH_TYPE_INSTANCE_PRINCIPAL = "instance_principal"
OCI_AUTH_TYPE_CONFIG_FILE = "config_file"
VALID_OCI_AUTH_TYPES = [OCI_AUTH_TYPE_INSTANCE_PRINCIPAL, OCI_AUTH_TYPE_CONFIG_FILE]
DEFAULT_OCI_REGION = "us-ashburn-1"

MODEL_CAPABILITIES = ["TEXT_GENERATION", "TEXT_SUMMARIZATION", "TEXT_EMBEDDINGS", "CHAT"]


class OCIInferenceAdapter(OpenAIMixin):
    config: OCIConfig

    async def initialize(self) -> None:
        """Initialize and validate OCI configuration."""
        if self.config.oci_auth_type not in VALID_OCI_AUTH_TYPES:
            raise ValueError(
                f"Invalid OCI authentication type: {self.config.oci_auth_type}."
                f"Valid types are one of: {VALID_OCI_AUTH_TYPES}"
            )

        if not self.config.oci_compartment_id:
            raise ValueError("OCI_COMPARTMENT_OCID is a required parameter. Either set in env variable or config.")

    def get_base_url(self) -> str:
        region = self.config.oci_region or DEFAULT_OCI_REGION
        return f"https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/v1"

    def get_api_key(self) -> str | None:
        # OCI doesn't use API keys, it uses request signing
        return "<NOTUSED>"

    def get_extra_client_params(self) -> dict[str, Any]:
        """
        Get extra parameters for the AsyncOpenAI client, including OCI-specific auth and headers.
        """
        auth = self._get_auth()
        compartment_id = self.config.oci_compartment_id or ""

        return {
            "http_client": DefaultAsyncHttpxClient(
                auth=auth,
                headers={
                    "CompartmentId": compartment_id,
                },
            ),
        }

    def _get_oci_signer(self) -> oci.signer.AbstractBaseSigner | None:
        if self.config.oci_auth_type == OCI_AUTH_TYPE_INSTANCE_PRINCIPAL:
            return oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        return None

    def _get_oci_config(self) -> dict:
        if self.config.oci_auth_type == OCI_AUTH_TYPE_INSTANCE_PRINCIPAL:
            config = {"region": self.config.oci_region}
        elif self.config.oci_auth_type == OCI_AUTH_TYPE_CONFIG_FILE:
            config = oci.config.from_file(self.config.oci_config_file_path, self.config.oci_config_profile)
            if not config.get("region"):
                raise ValueError(
                    "Region not specified in config. Please specify in config or with OCI_REGION env variable."
                )

        return config

    def _get_auth(self) -> httpx.Auth:
        if self.config.oci_auth_type == OCI_AUTH_TYPE_INSTANCE_PRINCIPAL:
            return OciInstancePrincipalAuth()
        elif self.config.oci_auth_type == OCI_AUTH_TYPE_CONFIG_FILE:
            return OciUserPrincipalAuth(
                config_file=self.config.oci_config_file_path, profile_name=self.config.oci_config_profile
            )
        else:
            raise ValueError(f"Invalid OCI authentication type: {self.config.oci_auth_type}")

    async def list_provider_model_ids(self) -> Iterable[str]:
        """
        List available models from OCI Generative AI service.
        """
        oci_config = self._get_oci_config()
        oci_signer = self._get_oci_signer()
        compartment_id = self.config.oci_compartment_id or ""

        if oci_signer is None:
            client = GenerativeAiClient(config=oci_config)
        else:
            client = GenerativeAiClient(config=oci_config, signer=oci_signer)

        models: ModelCollection = client.list_models(
            compartment_id=compartment_id, capability=MODEL_CAPABILITIES, lifecycle_state="ACTIVE"
        ).data

        seen_models = set()
        model_ids = []
        for model in models.items:
            if model.time_deprecated or model.time_on_demand_retired:
                continue

            if "CHAT" not in model.capabilities or "FINE_TUNE" in model.capabilities:
                continue

            # Use display_name + model_type as the key to avoid conflicts
            model_key = (model.display_name, ModelType.llm)
            if model_key in seen_models:
                continue

            seen_models.add(model_key)
            model_ids.append(model.display_name)

        return model_ids

    async def openai_embeddings(self, params: OpenAIEmbeddingsRequestWithExtraBody) -> OpenAIEmbeddingsResponse:
        # The constructed url is a mask that hits OCI's "chat" action, which is not supported for embeddings.
        raise NotImplementedError("OCI Provider does not (currently) support embeddings")
