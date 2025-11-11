# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_stack.core.datatypes import BuildProvider, Provider, ToolGroupInput
from llama_stack.distributions.template import DistributionTemplate, RunConfigSettings
from llama_stack.providers.inline.files.localfs.config import LocalfsFilesImplConfig
from llama_stack.providers.inline.vector_io.faiss.config import FaissVectorIOConfig
from llama_stack.providers.remote.inference.oci.config import OCIConfig


def get_distribution_template(name: str = "oci") -> DistributionTemplate:
    providers = {
        "inference": [BuildProvider(provider_type="remote::oci")],
        "vector_io": [
            BuildProvider(provider_type="inline::faiss"),
            BuildProvider(provider_type="remote::chromadb"),
            BuildProvider(provider_type="remote::pgvector"),
        ],
        "safety": [BuildProvider(provider_type="inline::llama-guard")],
        "agents": [BuildProvider(provider_type="inline::meta-reference")],
        "eval": [BuildProvider(provider_type="inline::meta-reference")],
        "datasetio": [
            BuildProvider(provider_type="remote::huggingface"),
            BuildProvider(provider_type="inline::localfs"),
        ],
        "scoring": [
            BuildProvider(provider_type="inline::basic"),
            BuildProvider(provider_type="inline::llm-as-judge"),
            BuildProvider(provider_type="inline::braintrust"),
        ],
        "tool_runtime": [
            BuildProvider(provider_type="remote::brave-search"),
            BuildProvider(provider_type="remote::tavily-search"),
            BuildProvider(provider_type="inline::rag-runtime"),
            BuildProvider(provider_type="remote::model-context-protocol"),
        ],
        "files": [BuildProvider(provider_type="inline::localfs")],
    }

    inference_provider = Provider(
        provider_id="oci",
        provider_type="remote::oci",
        config=OCIConfig.sample_run_config(),
    )

    vector_io_provider = Provider(
        provider_id="faiss",
        provider_type="inline::faiss",
        config=FaissVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
    )

    files_provider = Provider(
        provider_id="meta-reference-files",
        provider_type="inline::localfs",
        config=LocalfsFilesImplConfig.sample_run_config(f"~/.llama/distributions/{name}"),
    )
    default_tool_groups = [
        ToolGroupInput(
            toolgroup_id="builtin::websearch",
            provider_id="tavily-search",
        ),
    ]

    return DistributionTemplate(
        name=name,
        distro_type="remote_hosted",
        description="Use Oracle Cloud Infrastructure (OCI) Generative AI for running LLM inference with scalable cloud services",
        container_image=None,
        template_path=Path(__file__).parent / "doc_template.md",
        providers=providers,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider],
                    "vector_io": [vector_io_provider],
                    "files": [files_provider],
                },
                default_tool_groups=default_tool_groups,
            ),
        },
        run_config_env_vars={
            "OCI_AUTH_TYPE": (
                "instance_principal",
                "OCI authentication type (instance_principal or config_file)",
            ),
            "OCI_REGION": (
                "",
                "OCI region (e.g., us-ashburn-1, us-chicago-1, us-phoenix-1, eu-frankfurt-1)",
            ),
            "OCI_COMPARTMENT_OCID": (
                "",
                "OCI compartment ID for the Generative AI service",
            ),
            "OCI_CONFIG_FILE_PATH": (
                "~/.oci/config",
                "OCI config file path (required if OCI_AUTH_TYPE is config_file)",
            ),
            "OCI_CLI_PROFILE": (
                "DEFAULT",
                "OCI CLI profile name to use from config file",
            ),
        },
    )
