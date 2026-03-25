# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_stack.core.datatypes import BuildProvider, ModelInput, Provider, ShieldInput
from llama_stack.distributions.template import DistributionTemplate, RunConfigSettings
from llama_stack.providers.inline.files.localfs.config import LocalfsFilesImplConfig
from llama_stack.providers.remote.inference.nvidia import NVIDIAConfig
from llama_stack.providers.remote.safety.nvidia import NVIDIASafetyConfig


def get_distribution_template(name: str = "nvidia") -> DistributionTemplate:
    providers = {
        "inference": [BuildProvider(provider_type="remote::nvidia")],
        "vector_io": [BuildProvider(provider_type="inline::faiss")],
        "safety": [BuildProvider(provider_type="remote::nvidia")],
        "agents": [BuildProvider(provider_type="inline::builtin")],
        "tool_runtime": [BuildProvider(provider_type="inline::file-search")],
        "files": [BuildProvider(provider_type="inline::localfs")],
    }

    inference_provider = Provider(
        provider_id="nvidia",
        provider_type="remote::nvidia",
        config=NVIDIAConfig.sample_run_config(),
    )
    safety_provider = Provider(
        provider_id="nvidia",
        provider_type="remote::nvidia",
        config=NVIDIASafetyConfig.sample_run_config(),
    )
    files_provider = Provider(
        provider_id="builtin-files",
        provider_type="inline::localfs",
        config=LocalfsFilesImplConfig.sample_run_config(f"~/.llama/distributions/{name}"),
    )
    inference_model = ModelInput(
        model_id="${env.INFERENCE_MODEL}",
        provider_id="nvidia",
    )
    safety_model = ModelInput(
        model_id="${env.SAFETY_MODEL}",
        provider_id="nvidia",
    )

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Use NVIDIA NIM for running LLM inference and safety",
        container_image=None,
        template_path=Path(__file__).parent / "doc_template.md",
        providers=providers,
        run_configs={
            "config.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider],
                    "files": [files_provider],
                },
            ),
            "run-with-safety.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [
                        inference_provider,
                        safety_provider,
                    ],
                    "files": [files_provider],
                },
                default_models=[inference_model, safety_model],
                default_shields=[ShieldInput(shield_id="${env.SAFETY_MODEL}", provider_id="nvidia")],
            ),
        },
        run_config_env_vars={
            "NVIDIA_API_KEY": (
                "",
                "NVIDIA API Key",
            ),
            "NVIDIA_APPEND_API_VERSION": (
                "True",
                "Whether to append the API version to the base_url",
            ),
            "GUARDRAILS_SERVICE_URL": (
                "http://0.0.0.0:7331",
                "URL for the NeMo Guardrails Service",
            ),
            "NVIDIA_GUARDRAILS_CONFIG_ID": (
                "self-check",
                "NVIDIA Guardrail Configuration ID",
            ),
            "INFERENCE_MODEL": (
                "Llama3.1-8B-Instruct",
                "Inference model",
            ),
            "SAFETY_MODEL": (
                "meta/llama-3.1-8b-instruct",
                "Name of the model to use for safety",
            ),
        },
    )
