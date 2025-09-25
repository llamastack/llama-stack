# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib.resources
import sys

from pydantic import BaseModel
from termcolor import cprint

from llama_stack.core.datatypes import BuildConfig
from llama_stack.core.distribution import get_provider_registry
from llama_stack.core.external import load_external_apis
from llama_stack.core.utils.exec import run_command
from llama_stack.core.utils.image_types import LlamaStackImageType
from llama_stack.distributions.template import DistributionTemplate
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import Api

log = get_logger(name=__name__, category="core")

# These are the dependencies needed by the distribution server.
# `llama-stack` is automatically installed by the installation script.
SERVER_DEPENDENCIES = [
    "aiosqlite",
    "fastapi",
    "fire",
    "httpx",
    "uvicorn",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp-proto-http",
]


class ApiInput(BaseModel):
    api: Api
    provider: str


def get_provider_dependencies(
    config: BuildConfig | DistributionTemplate,
) -> tuple[list[str], list[str], list[str]]:
    """Get normal and special dependencies from provider configuration."""
    if isinstance(config, DistributionTemplate):
        config = config.build_config()

    providers = config.distribution_spec.providers
    additional_pip_packages = config.additional_pip_packages

    deps = []
    external_provider_deps = []
    registry = get_provider_registry(config)
    for api_str, provider_or_providers in providers.items():
        providers_for_api = registry[Api(api_str)]

        providers = provider_or_providers if isinstance(provider_or_providers, list) else [provider_or_providers]

        for provider in providers:
            # Providers from BuildConfig and RunConfig are subtly different - not great
            provider_type = provider if isinstance(provider, str) else provider.provider_type

            if provider_type not in providers_for_api:
                raise ValueError(f"Provider `{provider}` is not available for API `{api_str}`")

            provider_spec = providers_for_api[provider_type]
            if hasattr(provider_spec, "is_external") and provider_spec.is_external:
                # this ensures we install the top level module for our external providers
                if provider_spec.module:
                    if isinstance(provider_spec.module, str):
                        external_provider_deps.append(provider_spec.module)
                    else:
                        external_provider_deps.extend(provider_spec.module)
            if hasattr(provider_spec, "pip_packages"):
                deps.extend(provider_spec.pip_packages)
            if hasattr(provider_spec, "container_image") and provider_spec.container_image:
                raise ValueError("A stack's dependencies cannot have a container image")

    normal_deps = []
    special_deps = []
    for package in deps:
        if any(f in package for f in ["--no-deps", "--index-url", "--extra-index-url"]):
            special_deps.append(package)
        else:
            normal_deps.append(package)

    normal_deps.extend(additional_pip_packages or [])

    return list(set(normal_deps)), list(set(special_deps)), list(set(external_provider_deps))


def print_pip_install_help(config: BuildConfig):
    normal_deps, special_deps, _ = get_provider_dependencies(config)

    cprint(
        f"Please install needed dependencies using the following commands:\n\nuv pip install {' '.join(normal_deps)}",
        color="yellow",
        file=sys.stderr,
    )
    for special_dep in special_deps:
        cprint(f"uv pip install {special_dep}", color="yellow", file=sys.stderr)
    print()


def build_image(
    build_config: BuildConfig,
    image_name: str,
    distro_or_config: str,
    run_config: str | None = None,
):
    container_base = build_config.distribution_spec.container_image or "python:3.12-slim"

    normal_deps, special_deps, external_provider_deps = get_provider_dependencies(build_config)
    normal_deps += SERVER_DEPENDENCIES
    if build_config.external_apis_dir:
        external_apis = load_external_apis(build_config)
        if external_apis:
            for _, api_spec in external_apis.items():
                normal_deps.extend(api_spec.pip_packages)

    if build_config.image_type == LlamaStackImageType.CONTAINER.value:
        script = str(importlib.resources.files("llama_stack") / "core/build_container.sh")
        args = [
            script,
            "--distro-or-config",
            distro_or_config,
            "--image-name",
            image_name,
            "--container-base",
            container_base,
            "--normal-deps",
            " ".join(normal_deps),
        ]
        # When building from a config file (not a template), include the run config path in the
        # build arguments
        if run_config is not None:
            args.extend(["--run-config", run_config])
    else:
        script = str(importlib.resources.files("llama_stack") / "core/build_venv.sh")
        args = [
            script,
            "--env-name",
            str(image_name),
            "--normal-deps",
            " ".join(normal_deps),
        ]

    # Always pass both arguments, even if empty, to maintain consistent positional arguments
    if special_deps:
        args.extend(["--optional-deps", "#".join(special_deps)])
    if external_provider_deps:
        args.extend(
            ["--external-provider-deps", "#".join(external_provider_deps)]
        )  # the script will install external provider module, get its deps, and install those too.

    return_code = run_command(args)

    if return_code != 0:
        log.error(
            f"Failed to build target {image_name} with return code {return_code}",
        )

    return return_code
