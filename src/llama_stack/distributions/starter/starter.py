# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import Any

from llama_stack.core.datatypes import (
    BuildProvider,
    Provider,
    ProviderSpec,
    QualifiedModel,
    SafetyConfig,
    ShieldInput,
    ToolGroupInput,
    VectorStoresConfig,
)
from llama_stack.core.storage.datatypes import (
    InferenceStoreReference,
    KVStoreReference,
    SqlStoreReference,
)
from llama_stack.core.utils.dynamic import instantiate_class_type
from llama_stack.distributions.template import DistributionTemplate, RunConfigSettings
from llama_stack.providers.datatypes import RemoteProviderSpec
from llama_stack.providers.inline.files.localfs.config import LocalfsFilesImplConfig
from llama_stack.providers.inline.inference.sentence_transformers import (
    SentenceTransformersInferenceConfig,
)
from llama_stack.providers.inline.vector_io.faiss.config import FaissVectorIOConfig
from llama_stack.providers.inline.vector_io.milvus.config import MilvusVectorIOConfig
from llama_stack.providers.inline.vector_io.sqlite_vec.config import (
    SQLiteVectorIOConfig,
)
from llama_stack.providers.registry.inference import available_providers
from llama_stack.providers.remote.vector_io.chroma.config import ChromaVectorIOConfig
from llama_stack.providers.remote.vector_io.pgvector.config import (
    PGVectorVectorIOConfig,
)
from llama_stack.providers.remote.vector_io.qdrant.config import QdrantVectorIOConfig
from llama_stack.providers.remote.vector_io.weaviate.config import WeaviateVectorIOConfig
from llama_stack.providers.utils.kvstore.config import PostgresKVStoreConfig
from llama_stack.providers.utils.sqlstore.sqlstore import PostgresSqlStoreConfig


def _get_config_for_provider(provider_spec: ProviderSpec) -> dict[str, Any]:
    """Get configuration for a provider using its adapter's config class."""
    config_class = instantiate_class_type(provider_spec.config_class)

    if hasattr(config_class, "sample_run_config"):
        config: dict[str, Any] = config_class.sample_run_config()
        return config
    return {}


ENABLED_INFERENCE_PROVIDERS = [
    "ollama",
    "vllm",
    "tgi",
    "fireworks",
    "together",
    "gemini",
    "vertexai",
    "groq",
    "sambanova",
    "anthropic",
    "openai",
    "cerebras",
    "nvidia",
    "bedrock",
    "azure",
]

INFERENCE_PROVIDER_IDS = {
    "ollama": "${env.OLLAMA_URL:+ollama}",
    "vllm": "${env.VLLM_URL:+vllm}",
    "tgi": "${env.TGI_URL:+tgi}",
    "cerebras": "${env.CEREBRAS_API_KEY:+cerebras}",
    "nvidia": "${env.NVIDIA_API_KEY:+nvidia}",
    "vertexai": "${env.VERTEX_AI_PROJECT:+vertexai}",
    "azure": "${env.AZURE_API_KEY:+azure}",
}


def get_remote_inference_providers() -> list[Provider]:
    # Filter out inline providers and some others - the starter distro only exposes remote providers
    remote_providers = [
        provider
        for provider in available_providers()
        if isinstance(provider, RemoteProviderSpec) and provider.adapter_type in ENABLED_INFERENCE_PROVIDERS
    ]

    inference_providers = []
    for provider_spec in remote_providers:
        provider_type = provider_spec.adapter_type

        if provider_type in INFERENCE_PROVIDER_IDS:
            provider_id = INFERENCE_PROVIDER_IDS[provider_type]
        else:
            provider_id = provider_type.replace("-", "_").replace("::", "_")
        config = _get_config_for_provider(provider_spec)

        inference_providers.append(
            Provider(
                provider_id=provider_id,
                provider_type=f"remote::{provider_type}",
                config=config,
            )
        )
    return inference_providers


def get_distribution_template(name: str = "starter") -> DistributionTemplate:
    remote_inference_providers = get_remote_inference_providers()

    providers = {
        "inference": [BuildProvider(provider_type=p.provider_type, module=p.module) for p in remote_inference_providers]
        + [BuildProvider(provider_type="inline::sentence-transformers")],
        "vector_io": [
            BuildProvider(provider_type="inline::faiss"),
            BuildProvider(provider_type="inline::sqlite-vec"),
            BuildProvider(provider_type="inline::milvus"),
            BuildProvider(provider_type="remote::chromadb"),
            BuildProvider(provider_type="remote::pgvector"),
            BuildProvider(provider_type="remote::qdrant"),
            BuildProvider(provider_type="remote::weaviate"),
        ],
        "files": [BuildProvider(provider_type="inline::localfs")],
        "safety": [
            BuildProvider(provider_type="inline::llama-guard"),
            BuildProvider(provider_type="inline::code-scanner"),
        ],
        "agents": [BuildProvider(provider_type="inline::meta-reference")],
        "post_training": [BuildProvider(provider_type="inline::torchtune-cpu")],
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
        "batches": [
            BuildProvider(provider_type="inline::reference"),
        ],
    }
    files_provider = Provider(
        provider_id="meta-reference-files",
        provider_type="inline::localfs",
        config=LocalfsFilesImplConfig.sample_run_config(f"~/.llama/distributions/{name}"),
    )
    embedding_provider = Provider(
        provider_id="sentence-transformers",
        provider_type="inline::sentence-transformers",
        config=SentenceTransformersInferenceConfig.sample_run_config(),
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
    default_shields = [
        # if the
        ShieldInput(
            shield_id="llama-guard",
            provider_id="${env.SAFETY_MODEL:+llama-guard}",
            provider_shield_id="${env.SAFETY_MODEL:=}",
        ),
        ShieldInput(
            shield_id="code-scanner",
            provider_id="${env.CODE_SCANNER_MODEL:+code-scanner}",
            provider_shield_id="${env.CODE_SCANNER_MODEL:=}",
        ),
    ]
    postgres_config = PostgresSqlStoreConfig.sample_run_config()
    default_overrides = {
        "inference": remote_inference_providers + [embedding_provider],
        "vector_io": [
            Provider(
                provider_id="faiss",
                provider_type="inline::faiss",
                config=FaissVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
            ),
            Provider(
                provider_id="sqlite-vec",
                provider_type="inline::sqlite-vec",
                config=SQLiteVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
            ),
            Provider(
                provider_id="${env.MILVUS_URL:+milvus}",
                provider_type="inline::milvus",
                config=MilvusVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
            ),
            Provider(
                provider_id="${env.CHROMADB_URL:+chromadb}",
                provider_type="remote::chromadb",
                config=ChromaVectorIOConfig.sample_run_config(
                    f"~/.llama/distributions/{name}/",
                    url="${env.CHROMADB_URL:=}",
                ),
            ),
            Provider(
                provider_id="${env.PGVECTOR_DB:+pgvector}",
                provider_type="remote::pgvector",
                config=PGVectorVectorIOConfig.sample_run_config(
                    f"~/.llama/distributions/{name}",
                    db="${env.PGVECTOR_DB:=}",
                    user="${env.PGVECTOR_USER:=}",
                    password="${env.PGVECTOR_PASSWORD:=}",
                ),
            ),
            Provider(
                provider_id="${env.QDRANT_URL:+qdrant}",
                provider_type="remote::qdrant",
                config=QdrantVectorIOConfig.sample_run_config(
                    f"~/.llama/distributions/{name}",
                    url="${env.QDRANT_URL:=}",
                ),
            ),
            Provider(
                provider_id="${env.WEAVIATE_CLUSTER_URL:+weaviate}",
                provider_type="remote::weaviate",
                config=WeaviateVectorIOConfig.sample_run_config(
                    f"~/.llama/distributions/{name}",
                    cluster_url="${env.WEAVIATE_CLUSTER_URL:=}",
                ),
            ),
        ],
        "files": [files_provider],
    }

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Quick start template for running Llama Stack with several popular providers. This distribution is intended for CPU-only environments.",
        container_image=None,
        template_path=None,
        providers=providers,
        additional_pip_packages=list(set(PostgresSqlStoreConfig.pip_packages() + PostgresKVStoreConfig.pip_packages())),
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides=default_overrides,
                default_models=[],
                default_tool_groups=default_tool_groups,
                default_shields=default_shields,
                vector_stores_config=VectorStoresConfig(
                    default_provider_id="faiss",
                    default_embedding_model=QualifiedModel(
                        provider_id="sentence-transformers",
                        model_id="nomic-ai/nomic-embed-text-v1.5",
                    ),
                ),
                safety_config=SafetyConfig(
                    default_shield_id="llama-guard",
                ),
            ),
            "run-with-postgres-store.yaml": RunConfigSettings(
                provider_overrides={
                    **default_overrides,
                    "agents": [
                        Provider(
                            provider_id="meta-reference",
                            provider_type="inline::meta-reference",
                            config=dict(
                                persistence_store=postgres_config,
                                responses_store=postgres_config,
                            ),
                        )
                    ],
                    "batches": [
                        Provider(
                            provider_id="reference",
                            provider_type="inline::reference",
                            config=dict(
                                kvstore=KVStoreReference(
                                    backend="kv_postgres",
                                    namespace="batches",
                                ).model_dump(exclude_none=True),
                            ),
                        )
                    ],
                },
                storage_backends={
                    "kv_postgres": PostgresKVStoreConfig.sample_run_config(),
                    "sql_postgres": postgres_config,
                },
                storage_stores={
                    "metadata": KVStoreReference(
                        backend="kv_postgres",
                        namespace="registry",
                    ).model_dump(exclude_none=True),
                    "inference": InferenceStoreReference(
                        backend="sql_postgres",
                        table_name="inference_store",
                    ).model_dump(exclude_none=True),
                    "conversations": SqlStoreReference(
                        backend="sql_postgres",
                        table_name="openai_conversations",
                    ).model_dump(exclude_none=True),
                    "prompts": KVStoreReference(
                        backend="kv_postgres",
                        namespace="prompts",
                    ).model_dump(exclude_none=True),
                },
            ),
        },
        run_config_env_vars={
            "LLAMA_STACK_PORT": (
                "8321",
                "Port for the Llama Stack distribution server",
            ),
            "FIREWORKS_API_KEY": (
                "",
                "Fireworks API Key",
            ),
            "OPENAI_API_KEY": (
                "",
                "OpenAI API Key",
            ),
            "GROQ_API_KEY": (
                "",
                "Groq API Key",
            ),
            "ANTHROPIC_API_KEY": (
                "",
                "Anthropic API Key",
            ),
            "GEMINI_API_KEY": (
                "",
                "Gemini API Key",
            ),
            "VERTEX_AI_PROJECT": (
                "",
                "Google Cloud Project ID for Vertex AI",
            ),
            "VERTEX_AI_LOCATION": (
                "us-central1",
                "Google Cloud Location for Vertex AI",
            ),
            "SAMBANOVA_API_KEY": (
                "",
                "SambaNova API Key",
            ),
            "VLLM_URL": (
                "http://localhost:8000/v1",
                "vLLM URL",
            ),
            "VLLM_INFERENCE_MODEL": (
                "",
                "Optional vLLM Inference Model to register on startup",
            ),
            "OLLAMA_URL": (
                "http://localhost:11434",
                "Ollama URL",
            ),
            "AZURE_API_KEY": (
                "",
                "Azure API Key",
            ),
            "AZURE_API_BASE": (
                "",
                "Azure API Base",
            ),
            "AZURE_API_VERSION": (
                "",
                "Azure API Version",
            ),
            "AZURE_API_TYPE": (
                "azure",
                "Azure API Type",
            ),
        },
    )
