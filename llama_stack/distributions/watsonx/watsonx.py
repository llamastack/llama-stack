# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_stack.apis.models import ModelType
from llama_stack.core.datatypes import BuildProvider, ModelInput, Provider, ToolGroupInput
from llama_stack.distributions.template import DistributionTemplate, RunConfigSettings, get_model_registry
from llama_stack.providers.inline.inference.sentence_transformers import (
    SentenceTransformersInferenceConfig,
)
from llama_stack.providers.remote.inference.watsonx import WatsonXConfig
from llama_stack.providers.remote.inference.watsonx.models import MODEL_ENTRIES


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": [
            BuildProvider(provider_type="remote::watsonx"),
            BuildProvider(provider_type="inline::sentence-transformers"),
        ],
        "vector_io": [BuildProvider(provider_type="inline::faiss")],
        "safety": [BuildProvider(provider_type="inline::llama-guard")],
        "agents": [BuildProvider(provider_type="inline::meta-reference")],
        "telemetry": [BuildProvider(provider_type="inline::meta-reference")],
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
    }

    inference_provider = Provider(
        provider_id="watsonx",
        provider_type="remote::watsonx",
        config=WatsonXConfig.sample_run_config(),
    )

    embedding_provider = Provider(
        provider_id="sentence-transformers",
        provider_type="inline::sentence-transformers",
        config=SentenceTransformersInferenceConfig.sample_run_config(),
    )

    available_models = {
        "watsonx": MODEL_ENTRIES,
    }
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

    embedding_model = ModelInput(
        model_id="all-MiniLM-L6-v2",
        provider_id="sentence-transformers",
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 384,
        },
    )

    default_models, _ = get_model_registry(available_models)
    return DistributionTemplate(
        name="watsonx",
        distro_type="remote_hosted",
        description="Use watsonx for running LLM inference",
        container_image=None,
        template_path=Path(__file__).parent / "doc_template.md",
        providers=providers,
        available_models_by_provider=available_models,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider, embedding_provider],
                },
                default_models=default_models + [embedding_model],
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
