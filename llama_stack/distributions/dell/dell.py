# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.models import ModelType
from llama_stack.core.datatypes import (
    BuildProvider,
    ModelInput,
    Provider,
    ShieldInput,
    ToolGroupInput,
)
from llama_stack.distributions.template import DistributionTemplate, RunConfigSettings
from llama_stack.providers.inline.inference.sentence_transformers import (
    SentenceTransformersInferenceConfig,
)
from llama_stack.providers.remote.vector_io.chroma import ChromaVectorIOConfig


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": [
            BuildProvider(provider_type="remote::tgi"),
            BuildProvider(provider_type="inline::sentence-transformers"),
        ],
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
        ],
    }
    name = "dell"
    inference_provider = Provider(
        provider_id="tgi0",
        provider_type="remote::tgi",
        config={
            "url": "${env.DEH_URL}",
        },
    )
    safety_inference_provider = Provider(
        provider_id="tgi1",
        provider_type="remote::tgi",
        config={
            "url": "${env.DEH_SAFETY_URL}",
        },
    )
    embedding_provider = Provider(
        provider_id="sentence-transformers",
        provider_type="inline::sentence-transformers",
        config=SentenceTransformersInferenceConfig.sample_run_config(),
    )
    chromadb_provider = Provider(
        provider_id="chromadb",
        provider_type="remote::chromadb",
        config=ChromaVectorIOConfig.sample_run_config(
            f"~/.llama/distributions/{name}/",
            url="${env.CHROMADB_URL:=}",
        ),
    )

    inference_model = ModelInput(
        model_id="${env.INFERENCE_MODEL}",
        provider_id="tgi0",
    )
    safety_model = ModelInput(
        model_id="${env.SAFETY_MODEL}",
        provider_id="tgi1",
    )
    embedding_model = ModelInput(
        model_id="nomic-embed-text-v1.5",
        provider_id="sentence-transformers",
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 768,
        },
    )
    default_tool_groups = [
        ToolGroupInput(
            toolgroup_id="builtin::websearch",
            provider_id="brave-search",
        ),
        ToolGroupInput(
            toolgroup_id="builtin::rag",
            provider_id="rag-runtime",
        ),
    ]

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Dell's distribution of Llama Stack. TGI inference via Dell's custom container",
        container_image=None,
        providers=providers,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider, embedding_provider],
                    "vector_io": [chromadb_provider],
                },
                default_models=[inference_model, embedding_model],
                default_tool_groups=default_tool_groups,
            ),
            "run-with-safety.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [
                        inference_provider,
                        safety_inference_provider,
                        embedding_provider,
                    ],
                    "vector_io": [chromadb_provider],
                },
                default_models=[inference_model, safety_model, embedding_model],
                default_shields=[ShieldInput(shield_id="${env.SAFETY_MODEL}")],
                default_tool_groups=default_tool_groups,
            ),
        },
        run_config_env_vars={
            "DEH_URL": (
                "http://0.0.0.0:8181",
                "URL for the Dell inference server",
            ),
            "DEH_SAFETY_URL": (
                "http://0.0.0.0:8282",
                "URL for the Dell safety inference server",
            ),
            "CHROMA_URL": (
                "http://localhost:6601",
                "URL for the Chroma server",
            ),
            "INFERENCE_MODEL": (
                "meta-llama/Llama-3.2-3B-Instruct",
                "Inference model loaded into the TGI server",
            ),
            "SAFETY_MODEL": (
                "meta-llama/Llama-Guard-3-1B",
                "Name of the safety (Llama-Guard) model to use",
            ),
        },
    )
