# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import sys
from pathlib import Path

import yaml
from termcolor import cprint

from llama_stack.cli.stack.utils import ImageType
from llama_stack.core.build import get_provider_dependencies
from llama_stack.core.datatypes import (
    BuildConfig,
    BuildProvider,
    DistributionSpec,
)
from llama_stack.core.distribution import get_provider_registry
from llama_stack.core.stack import replace_env_vars
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import Api

TEMPLATES_PATH = Path(__file__).parent.parent.parent / "templates"

logger = get_logger(name=__name__, category="cli")


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


def format_output_deps_only(
    normal_deps: list[str],
    special_deps: list[str],
    external_deps: list[str],
    uv: bool = False,
) -> str:
    """Format dependencies as a list."""
    lines = []

    uv_str = ""
    if uv:
        uv_str = "uv pip install "

    # Quote deps with commas
    quoted_normal_deps = [quote_if_needed(dep) for dep in normal_deps]
    lines.append(f"{uv_str}{' '.join(quoted_normal_deps)}")

    for special_dep in special_deps:
        lines.append(f"{uv_str}{quote_special_dep(special_dep)}")

    for external_dep in external_deps:
        lines.append(f"{uv_str}{quote_special_dep(external_dep)}")

    return "\n".join(lines)


def run_stack_list_deps_command(args: argparse.Namespace) -> None:
    if args.config:
        try:
            from llama_stack.core.utils.config_resolution import Mode, resolve_config_or_distro

            config_file = resolve_config_or_distro(args.config, Mode.BUILD)
        except ValueError as e:
            cprint(
                f"Could not parse config file {args.config}: {e}",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)
        if config_file:
            with open(config_file) as f:
                try:
                    contents = yaml.safe_load(f)
                    contents = replace_env_vars(contents)
                    build_config = BuildConfig(**contents)
                    build_config.image_type = "venv"
                except Exception as e:
                    cprint(
                        f"Could not parse config file {config_file}: {e}",
                        color="red",
                        file=sys.stderr,
                    )
                    sys.exit(1)
    elif args.providers:
        provider_list: dict[str, list[BuildProvider]] = dict()
        for api_provider in args.providers.split(","):
            if "=" not in api_provider:
                cprint(
                    "Could not parse `--providers`. Please ensure the list is in the format api1=provider1,api2=provider2",
                    color="red",
                    file=sys.stderr,
                )
                sys.exit(1)
            api, provider_type = api_provider.split("=")
            providers_for_api = get_provider_registry().get(Api(api), None)
            if providers_for_api is None:
                cprint(
                    f"{api} is not a valid API.",
                    color="red",
                    file=sys.stderr,
                )
                sys.exit(1)
            if provider_type in providers_for_api:
                provider = BuildProvider(
                    provider_type=provider_type,
                    module=None,
                )
                provider_list.setdefault(api, []).append(provider)
            else:
                cprint(
                    f"{provider_type} is not a valid provider for the {api} API.",
                    color="red",
                    file=sys.stderr,
                )
                sys.exit(1)
        distribution_spec = DistributionSpec(
            providers=provider_list,
            description=",".join(args.providers),
        )
        build_config = BuildConfig(image_type=ImageType.VENV.value, distribution_spec=distribution_spec)

    normal_deps, special_deps, external_provider_dependencies = get_provider_dependencies(build_config)
    normal_deps += SERVER_DEPENDENCIES

    # Add external API dependencies
    if build_config.external_apis_dir:
        from llama_stack.core.external import load_external_apis

        external_apis = load_external_apis(build_config)
        if external_apis:
            for _, api_spec in external_apis.items():
                normal_deps.extend(api_spec.pip_packages)

    # Format and output based on requested format
    output = format_output_deps_only(
        normal_deps=normal_deps,
        special_deps=special_deps,
        external_deps=external_provider_dependencies,
        uv=args.format == "uv",
    )

    print(output)


def quote_if_needed(dep):
    # Add quotes if the dependency contains special characters that need escaping in shell
    # This includes: commas, comparison operators (<, >, <=, >=, ==, !=)
    needs_quoting = any(char in dep for char in [",", "<", ">", "="])
    return f"'{dep}'" if needs_quoting else dep


def quote_special_dep(dep_string):
    """
    Quote individual packages in a special dependency string.
    Special deps may contain multiple packages and flags like --extra-index-url.
    We need to quote only the package specs that contain special characters.
    """
    parts = dep_string.split()
    quoted_parts = []

    for part in parts:
        # Don't quote flags (they start with -)
        if part.startswith("-"):
            quoted_parts.append(part)
        else:
            # Quote package specs that need it
            quoted_parts.append(quote_if_needed(part))

    return " ".join(quoted_parts)
