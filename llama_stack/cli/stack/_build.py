# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import importlib.resources
import json
import os
import shutil
import sys
import textwrap
from functools import lru_cache
from importlib.abc import Traversable
from pathlib import Path

import yaml
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.validation import Validator
from termcolor import colored, cprint

from llama_stack.cli.stack.utils import ImageType
from llama_stack.cli.table import print_table
from llama_stack.core.build import (
    SERVER_DEPENDENCIES,
    build_image,
    get_provider_dependencies,
)
from llama_stack.core.configure import parse_and_maybe_upgrade_config
from llama_stack.core.datatypes import (
    BuildConfig,
    BuildProvider,
    DistributionSpec,
    Provider,
    StackRunConfig,
)
from llama_stack.core.distribution import get_provider_registry
from llama_stack.core.external import load_external_apis
from llama_stack.core.resolver import InvalidProviderError
from llama_stack.core.stack import replace_env_vars
from llama_stack.core.utils.config_dirs import DISTRIBS_BASE_DIR, EXTERNAL_PROVIDERS_DIR
from llama_stack.core.utils.dynamic import instantiate_class_type
from llama_stack.core.utils.exec import formulate_run_args, run_command
from llama_stack.core.utils.image_types import LlamaStackImageType
from llama_stack.providers.datatypes import Api
from llama_stack.providers.utils.sqlstore.sqlstore import SqliteSqlStoreConfig

DISTRIBS_PATH = Path(__file__).parent.parent.parent / "distributions"


@lru_cache
def available_distros_specs() -> dict[str, BuildConfig]:
    import yaml

    distro_specs = {}
    for p in DISTRIBS_PATH.rglob("*build.yaml"):
        distro_name = p.parent.name
        with open(p) as f:
            build_config = BuildConfig(**yaml.safe_load(f))
            distro_specs[distro_name] = build_config
    return distro_specs


def run_stack_build_command(args: argparse.Namespace) -> None:
    if args.list_distros:
        return _run_distro_list_cmd()

    if args.image_type == ImageType.VENV.value:
        current_venv = os.environ.get("VIRTUAL_ENV")
        image_name = args.image_name or current_venv
    else:
        image_name = args.image_name

    if args.template:
        cprint(
            "The --template argument is deprecated. Please use --distro instead.",
            color="red",
            file=sys.stderr,
        )
        distro_name = args.template
    else:
        distro_name = args.distribution

    if distro_name:
        available_distros = available_distros_specs()
        if distro_name not in available_distros:
            cprint(
                f"Could not find distribution {distro_name}. Please run `llama stack build --list-distros` to check out the available distributions",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)
        build_config = available_distros[distro_name]
        if args.image_type:
            build_config.image_type = args.image_type
        else:
            cprint(
                f"Please specify a image-type ({' | '.join(e.value for e in ImageType)}) for {distro_name}",
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
                    f"{provider} is not a valid provider for the {api} API.",
                    color="red",
                    file=sys.stderr,
                )
                sys.exit(1)
        distribution_spec = DistributionSpec(
            providers=provider_list,
            description=",".join(args.providers),
        )
        if not args.image_type:
            cprint(
                f"Please specify a image-type (container | venv) for {args.template}",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)

        build_config = BuildConfig(image_type=args.image_type, distribution_spec=distribution_spec)
    elif not args.config and not distro_name:
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

        image_name = f"llamastack-{name}"

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
                if args.image_type:
                    build_config.image_type = args.image_type
            except Exception as e:
                cprint(
                    f"Could not parse config file {args.config}: {e}",
                    color="red",
                    file=sys.stderr,
                )
                sys.exit(1)

    if args.print_deps_only:
        print(f"# Dependencies for {distro_name or args.config or image_name}")
        normal_deps, special_deps, external_provider_dependencies = get_provider_dependencies(build_config)
        normal_deps += SERVER_DEPENDENCIES
        print(f"uv pip install {' '.join(normal_deps)}")
        for special_dep in special_deps:
            print(f"uv pip install {special_dep}")
        for external_dep in external_provider_dependencies:
            print(f"uv pip install {external_dep}")
        return

    try:
        run_config = _run_stack_build_command_from_build_config(
            build_config,
            image_name=image_name,
            config_path=args.config,
            distro_name=distro_name,
        )

    except (Exception, RuntimeError) as exc:
        import traceback

        cprint(
            f"Error building stack: {exc}",
            color="red",
            file=sys.stderr,
        )
        cprint("Stack trace:", color="red", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    if run_config is None:
        cprint(
            "Run config path is empty",
            color="red",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.run:
        config_dict = yaml.safe_load(run_config.read_text())
        config = parse_and_maybe_upgrade_config(config_dict)
        if config.external_providers_dir and not config.external_providers_dir.exists():
            config.external_providers_dir.mkdir(exist_ok=True)
        run_args = formulate_run_args(args.image_type, image_name or config.image_name)
        run_args.extend([str(os.getenv("LLAMA_STACK_PORT", 8321)), "--config", str(run_config)])
        run_command(run_args)


def _generate_run_config(
    build_config: BuildConfig,
    build_dir: Path,
    image_name: str,
) -> Path:
    """
    Generate a run.yaml template file for user to edit from a build.yaml file
    """
    apis = list(build_config.distribution_spec.providers.keys())
    run_config = StackRunConfig(
        container_image=(image_name if build_config.image_type == LlamaStackImageType.CONTAINER.value else None),
        image_name=image_name,
        apis=apis,
        providers={},
        external_providers_dir=build_config.external_providers_dir
        if build_config.external_providers_dir
        else EXTERNAL_PROVIDERS_DIR,
    )
    if not run_config.inference_store:
        run_config.inference_store = SqliteSqlStoreConfig(
            **SqliteSqlStoreConfig.sample_run_config(
                __distro_dir__=(DISTRIBS_BASE_DIR / image_name).as_posix(), db_name="inference_store.db"
            )
        )
    # build providers dict
    provider_registry = get_provider_registry(build_config)
    for api in apis:
        run_config.providers[api] = []
        providers = build_config.distribution_spec.providers[api]

        for provider in providers:
            pid = provider.provider_type.split("::")[-1]

            p = provider_registry[Api(api)][provider.provider_type]
            if p.deprecation_error:
                raise InvalidProviderError(p.deprecation_error)

            try:
                config_type = instantiate_class_type(provider_registry[Api(api)][provider.provider_type].config_class)
            except (ModuleNotFoundError, ValueError) as exc:
                # HACK ALERT:
                # This code executes after building is done, the import cannot work since the
                # package is either available in the venv or container - not available on the host.
                # TODO: use a "is_external" flag in ProviderSpec to check if the provider is
                # external
                cprint(
                    f"Failed to import provider {provider.provider_type} for API {api} - assuming it's external, skipping: {exc}",
                    color="yellow",
                    file=sys.stderr,
                )
                # Set config_type to None to avoid UnboundLocalError
                config_type = None

            if config_type is not None and hasattr(config_type, "sample_run_config"):
                config = config_type.sample_run_config(__distro_dir__=f"~/.llama/distributions/{image_name}")
            else:
                config = {}

            p_spec = Provider(
                provider_id=pid,
                provider_type=provider.provider_type,
                config=config,
                module=provider.module,
            )
            run_config.providers[api].append(p_spec)

    run_config_file = build_dir / f"{image_name}-run.yaml"

    with open(run_config_file, "w") as f:
        to_write = json.loads(run_config.model_dump_json())
        f.write(yaml.dump(to_write, sort_keys=False))

    # Only print this message for non-container builds since it will be displayed before the
    # container is built
    # For non-container builds, the run.yaml is generated at the very end of the build process so it
    # makes sense to display this message
    if build_config.image_type != LlamaStackImageType.CONTAINER.value:
        cprint(f"You can now run your stack with `llama stack run {run_config_file}`", color="green", file=sys.stderr)
    return run_config_file


def _run_stack_build_command_from_build_config(
    build_config: BuildConfig,
    image_name: str | None = None,
    distro_name: str | None = None,
    config_path: str | None = None,
) -> Path | Traversable:
    image_name = image_name or build_config.image_name
    if build_config.image_type == LlamaStackImageType.CONTAINER.value:
        if distro_name:
            image_name = f"distribution-{distro_name}"
        else:
            if not image_name:
                raise ValueError("Please specify an image name when building a container image without a template")
    else:
        if not image_name and os.environ.get("UV_SYSTEM_PYTHON"):
            image_name = "__system__"
        if not image_name:
            raise ValueError("Please specify an image name when building a venv image")

    # At this point, image_name should be guaranteed to be a string
    if image_name is None:
        raise ValueError("image_name should not be None after validation")

    if distro_name:
        build_dir = DISTRIBS_BASE_DIR / distro_name
        build_file_path = build_dir / f"{distro_name}-build.yaml"
    else:
        if image_name is None:
            raise ValueError("image_name cannot be None")
        build_dir = DISTRIBS_BASE_DIR / image_name
        build_file_path = build_dir / f"{image_name}-build.yaml"

    os.makedirs(build_dir, exist_ok=True)
    run_config_file = None
    # Generate the run.yaml so it can be included in the container image with the proper entrypoint
    # Only do this if we're building a container image and we're not using a template
    if build_config.image_type == LlamaStackImageType.CONTAINER.value and not distro_name and config_path:
        cprint("Generating run.yaml file", color="yellow", file=sys.stderr)
        run_config_file = _generate_run_config(build_config, build_dir, image_name)

    with open(build_file_path, "w") as f:
        to_write = json.loads(build_config.model_dump_json(exclude_none=True))
        f.write(yaml.dump(to_write, sort_keys=False))

    # We first install the external APIs so that the build process can use them and discover the
    # providers dependencies
    if build_config.external_apis_dir:
        cprint("Installing external APIs", color="yellow", file=sys.stderr)
        external_apis = load_external_apis(build_config)
        if external_apis:
            # install the external APIs
            packages = []
            for _, api_spec in external_apis.items():
                if api_spec.pip_packages:
                    packages.extend(api_spec.pip_packages)
                    cprint(
                        f"Installing {api_spec.name} with pip packages {api_spec.pip_packages}",
                        color="yellow",
                        file=sys.stderr,
                    )
            return_code = run_command(["uv", "pip", "install", *packages])
            if return_code != 0:
                packages_str = ", ".join(packages)
                raise RuntimeError(
                    f"Failed to install external APIs packages: {packages_str} (return code: {return_code})"
                )

    return_code = build_image(
        build_config,
        image_name,
        distro_or_config=distro_name or config_path or str(build_file_path),
        run_config=run_config_file.as_posix() if run_config_file else None,
    )
    if return_code != 0:
        raise RuntimeError(f"Failed to build image {image_name}")

    if distro_name:
        # copy run.yaml from distribution to build_dir instead of generating it again
        distro_path = importlib.resources.files("llama_stack") / f"distributions/{distro_name}/run.yaml"
        run_config_file = build_dir / f"{distro_name}-run.yaml"

        with importlib.resources.as_file(distro_path) as path:
            shutil.copy(path, run_config_file)

        cprint("Build Successful!", color="green", file=sys.stderr)
        cprint(f"You can find the newly-built distribution here: {run_config_file}", color="blue", file=sys.stderr)
        if build_config.image_type == LlamaStackImageType.VENV:
            cprint(
                "You can run the new Llama Stack distro (after activating "
                + colored(image_name, "cyan")
                + ") via: "
                + colored(f"llama stack run {run_config_file}", "blue"),
                color="green",
                file=sys.stderr,
            )
        elif build_config.image_type == LlamaStackImageType.CONTAINER:
            cprint(
                "You can run the container with: "
                + colored(
                    f"docker run -p 8321:8321 -v ~/.llama:/root/.llama localhost/{image_name} --port 8321", "blue"
                ),
                color="green",
                file=sys.stderr,
            )
        return distro_path
    else:
        return _generate_run_config(build_config, build_dir, image_name)


def _run_distro_list_cmd() -> None:
    headers = [
        "Distribution Name",
        # "Providers",
        "Description",
    ]

    rows = []
    for distro_name, spec in available_distros_specs().items():
        rows.append(
            [
                distro_name,
                # json.dumps(spec.distribution_spec.providers, indent=2),
                spec.distribution_spec.description,
            ]
        )
    print_table(
        rows,
        headers,
        separate_rows=True,
    )
