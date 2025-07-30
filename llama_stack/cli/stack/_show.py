# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import yaml
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.validation import Validator
from termcolor import colored, cprint

from llama_stack.cli.stack.utils import ImageType, available_templates_specs, generate_run_config
from llama_stack.core.build import get_provider_dependencies
from llama_stack.core.datatypes import (
    BuildConfig,
    BuildProvider,
    DistributionSpec,
)
from llama_stack.core.distribution import get_provider_registry
from llama_stack.core.external import load_external_apis
from llama_stack.core.stack import replace_env_vars
from llama_stack.core.utils.config_dirs import DISTRIBS_BASE_DIR
from llama_stack.core.utils.exec import run_command
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


def format_output_json(
    name: Any | str | None,
    build_config: BuildConfig,
    normal_deps: list[str],
    special_deps: list[str],
    external_deps: list[str],
) -> str:
    """Format dependencies as JSON."""
    # Build APIs list with providers
    apis = []
    for api_str, provider_or_providers in build_config.distribution_spec.providers.items():
        providers = provider_or_providers if isinstance(provider_or_providers, list) else [provider_or_providers]
        for provider in providers:
            provider_type = provider if isinstance(provider, str) else provider.provider_type
            apis.append(
                {
                    "api": api_str,
                    "provider": provider_type,
                }
            )

    # Include external deps in normal pip_dependencies
    all_normal_deps = normal_deps + external_deps

    output = {
        "name": name,
        "description": build_config.distribution_spec.description or "",
        "apis": apis,
        "pip_dependencies": all_normal_deps,
        "special_pip_dependencies": special_deps,
    }

    return json.dumps(output, indent=2)


def format_output_plain(
    name: Any | str | None,
    normal_deps: list[str],
    special_deps: list[str],
    external_deps: list[str],
) -> str:
    """Format dependencies as plain shell commands."""
    lines = [f"# Dependencies for {name}"]

    # Quote deps with commas
    quoted_normal_deps = [quote_if_needed(dep) for dep in normal_deps]
    lines.append(f"uv pip install {' '.join(quoted_normal_deps)}")

    for special_dep in special_deps:
        lines.append(f"uv pip install {quote_if_needed(special_dep)}")

    for external_dep in external_deps:
        lines.append(f"uv pip install {quote_if_needed(external_dep)}")

    return "\n".join(lines)


def run_stack_show_command(args: argparse.Namespace) -> None:
    env_name = args.env_name

    if args.distro:
        available_templates = available_templates_specs()
        if args.distro not in available_templates:
            cprint(
                f"Could not find template {args.distro}. Please run `llama stack show --list-distros` to check out the available templates",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)
        build_config = available_templates[args.distro]
        # always venv, conda is gone and container is separate.
        build_config.image_type = ImageType.VENV.value
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
    elif not args.config and not args.distro:
        name = prompt(
            "> Enter a name for your Llama Stack (e.g. my-local-stack): ",
            validator=Validator.from_callable(
                lambda x: len(x) > 0,
                error_message="Name cannot be empty, please enter a name",
            ),
        )

        image_type = prompt(
            "> Enter the image type you want your Llama Stack to be built as (use <TAB> to see options): ",
            completer=WordCompleter([e.value for e in ImageType]),
            complete_while_typing=True,
            validator=Validator.from_callable(
                lambda x: x in [e.value for e in ImageType],
                error_message="Invalid image type. Use <TAB> to see options",
            ),
        )

        env_name = f"llamastack-{name}"

        cprint(
            textwrap.dedent(
                """
            Llama Stack is composed of several APIs working together. Let's select
            the provider types (implementations) you want to use for these APIs.
            """,
            ),
            color="green",
            file=sys.stderr,
        )

        cprint("Tip: use <TAB> to see options for the providers.\n", color="green", file=sys.stderr)

        providers: dict[str, list[BuildProvider]] = dict()
        for api, providers_for_api in get_provider_registry().items():
            available_providers = [x for x in providers_for_api.keys() if x not in ("remote", "remote::sample")]
            if not available_providers:
                continue
            api_provider = prompt(
                f"> Enter provider for API {api.value}: ",
                completer=WordCompleter(available_providers),
                complete_while_typing=True,
                validator=Validator.from_callable(
                    lambda x: x in available_providers,  # noqa: B023 - see https://github.com/astral-sh/ruff/issues/7847
                    error_message="Invalid provider, use <TAB> to see options",
                ),
            )

            string_providers = api_provider.split(" ")

            for provider in string_providers:
                providers.setdefault(api.value, []).append(BuildProvider(provider_type=provider))

        description = prompt(
            "\n > (Optional) Enter a short description for your Llama Stack: ",
            default="",
        )

        distribution_spec = DistributionSpec(
            providers=providers,
            description=description,
        )

        build_config = BuildConfig(image_type=image_type, distribution_spec=distribution_spec)
    else:
        with open(args.config) as f:
            try:
                contents = yaml.safe_load(f)
                contents = replace_env_vars(contents)
                build_config = BuildConfig(**contents)
                build_config.image_type = "venv"
            except Exception as e:
                cprint(
                    f"Could not parse config file {args.config}: {e}",
                    color="red",
                    file=sys.stderr,
                )
                sys.exit(1)

    name = env_name or args.distro or args.config  # the config should be the last option if `--env-name` was specified
    normal_deps, special_deps, external_provider_dependencies = get_provider_dependencies(build_config)
    normal_deps += SERVER_DEPENDENCIES

    # Format and output based on requested format
    if args.format == "json":
        output = format_output_json(
            name=name,
            build_config=build_config,
            normal_deps=normal_deps,
            special_deps=special_deps,
            external_deps=external_provider_dependencies,
        )
    else:
        output = format_output_plain(
            name=name,
            normal_deps=normal_deps,
            special_deps=special_deps,
            external_deps=external_provider_dependencies,
        )

    print(output)


def quote_if_needed(dep):
    # Add quotes if the dependency contains a comma (likely version specifier)
    return f"'{dep}'" if "," in dep else dep
