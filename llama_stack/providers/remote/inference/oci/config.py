# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Any

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


class OCIProviderDataValidator(BaseModel):
    oci_auth_type: str = Field(
        description="OCI authentication type (must be one of: instance_principal, config_file)",
    )
    oci_private_key: str | None = Field(
        description="OCI private key for authentication",
    )
    oci_config_file_path: str | None = Field(
        default=None,
        description="OCI config file path (required if oci_auth_type is config_file)",
    )
    oci_config_profile: str = Field(
        description="OCI config profile (required if oci_auth_type is config_file)",
    )
    oci_region: str | None = Field(
        default=None,
        description="OCI region (e.g., us-ashburn-1)",
    )
    oci_compartment_id: str | None = Field(
        default=None,
        description="OCI compartment ID for the Generative AI service",
    )
    oci_user_ocid: str | None = Field(
        default=None,
        description="OCI user OCID for authentication",
    )
    oci_tenancy_ocid: str | None = Field(
        default=None,
        description="OCI tenancy OCID for authentication",
    )
    oci_fingerprint: str | None = Field(
        default=None,
        description="OCI API key fingerprint for authentication",
    )
    oci_serving_mode: str = Field(
        default="ON_DEMAND",
        description="OCI serving mode (must be one of: ON_DEMAND, DEDICATED)",
    )


@json_schema_type
class OCIConfig(BaseModel):
    oci_auth_type: str = Field(
        description="OCI authentication type (must be one of: instance_principal, config_file)",
        default_factory=lambda: os.getenv("OCI_AUTH_TYPE", "instance_principal"),
    )
    oci_config_file_path: str = Field(
        default_factory=lambda: os.getenv("OCI_CONFIG_FILE_PATH", "~/.oci/config"),
        description="OCI config file path (required if oci_auth_type is config_file)",
    )
    oci_config_profile: str = Field(
        default_factory=lambda: os.getenv("OCI_CLI_PROFILE", "DEFAULT"),
        description="OCI config profile (required if oci_auth_type is config_file)",
    )
    oci_region: str | None = Field(
        default_factory=lambda: os.getenv("OCI_REGION"),
        description="OCI region (e.g., us-ashburn-1)",
    )
    oci_compartment_id: str | None = Field(
        default_factory=lambda: os.getenv("OCI_COMPARTMENT_OCID"),
        description="OCI compartment ID for the Generative AI service",
    )
    oci_user_ocid: str | None = Field(
        default_factory=lambda: os.getenv("OCI_USER_OCID"),
        description="OCI user OCID for authentication",
    )
    oci_tenancy_ocid: str | None = Field(
        default_factory=lambda: os.getenv("OCI_TENANCY_OCID"),
        description="OCI tenancy OCID for authentication",
    )
    oci_fingerprint: str | None = Field(
        default_factory=lambda: os.getenv("OCI_FINGERPRINT"),
        description="OCI API key fingerprint for authentication",
    )
    oci_private_key: str | None = Field(
        description="OCI private key for authentication",
    )
    oci_serving_mode: str = Field(
        default_factory=lambda: os.getenv("OCI_SERVING_MODE", "ON_DEMAND"),
        description="OCI serving mode (must be one of: ON_DEMAND, DEDICATED)",
    )

    @classmethod
    def sample_run_config(
        cls,
        oci_auth_type: str = "${env.OCI_AUTH_TYPE:=instance_principal}",
        oci_config_file_path: str = "${env.OCI_CONFIG_FILE_PATH:=~/.oci/config}",
        oci_config_profile: str = "${env.OCI_CLI_PROFILE:=DEFAULT}",
        oci_region: str = "${env.OCI_REGION:=us-ashburn-1}",
        oci_compartment_id: str = "${env.OCI_COMPARTMENT_OCID:=}",
        oci_serving_mode: str = "${env.OCI_SERVING_MODE:=ON_DEMAND}",
        oci_user_ocid: str = "${env.OCI_USER_OCID:=}",
        oci_tenancy_ocid: str = "${env.OCI_TENANCY_OCID:=}",
        oci_fingerprint: str = "${env.OCI_FINGERPRINT:=}",
        oci_private_key: str = "${env.OCI_PRIVATE_KEY:=}",
        **kwargs,
    ) -> dict[str, Any]:
        return {
            "oci_auth_type": oci_auth_type,
            "oci_config_file_path": oci_config_file_path,
            "oci_config_profile": oci_config_profile,
            "oci_region": oci_region,
            "oci_compartment_id": oci_compartment_id,
            "oci_serving_mode": oci_serving_mode,
            "oci_user_ocid": oci_user_ocid,
            "oci_tenancy_ocid": oci_tenancy_ocid,
            "oci_fingerprint": oci_fingerprint,
            "oci_private_key": oci_private_key,
        }
