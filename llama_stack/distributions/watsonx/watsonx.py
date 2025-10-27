# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.core.datatypes import BuildProvider, Provider, ToolGroupInput
from llama_stack.distributions.template import DistributionTemplate, RunConfigSettings
from llama_stack.providers.inline.files.localfs.config import LocalfsFilesImplConfig
from llama_stack.providers.remote.inference.watsonx import WatsonXConfig


def get_distribution_template(name: str = "watsonx") -> DistributionTemplate:
    providers = {
        "inference": [
            BuildProvider(provider_type="remote::watsonx"),
            BuildProvider(provider_type="inline::sentence-transformers"),
        ],
        "vector_io": [BuildProvider(provider_type="inline::faiss")],
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
        provider_id="watsonx",
        provider_type="remote::watsonx",
        config=WatsonXConfig.sample_run_config(),
    )

    default_tool_groups = [
        ToolGroupInput(
            toolgroup_id="builtin::websearch",
            provider_id="tavily-search",
        ),
        ToolGroupInput(
            toolgroup_id="builtin::rag",
            provider_id="rag-runtime",
        ),
    ]

    files_provider = Provider(
        provider_id="meta-reference-files",
        provider_type="inline::localfs",
        config=LocalfsFilesImplConfig.sample_run_config(f"~/.llama/distributions/{name}"),
    )
    return DistributionTemplate(
        name=name,
        distro_type="remote_hosted",
        description="Use watsonx for running LLM inference",
        container_image=None,
        template_path=None,
        providers=providers,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider],
                    "files": [files_provider],
                },
                default_models=[],
                default_tool_groups=default_tool_groups,
            ),
        },
        run_config_env_vars={
            "LLAMASTACK_PORT": (
                "5001",
                "Port for the Llama Stack distribution server",
            ),
            "WATSONX_API_KEY": (
                "",
                "watsonx API Key",
            ),
            "WATSONX_PROJECT_ID": (
                "",
                "watsonx Project ID",
            ),
        },
    )
