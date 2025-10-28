# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import sys
from enum import Enum
from functools import lru_cache
from pathlib import Path

import yaml
from termcolor import cprint

from llama_stack.core.datatypes import (
    BuildConfig,
    Provider,
    StackRunConfig,
    StorageConfig,
)
from llama_stack.core.distribution import get_provider_registry
from llama_stack.core.resolver import InvalidProviderError
from llama_stack.core.storage.datatypes import (
    InferenceStoreReference,
    KVStoreReference,
    ServerStoresConfig,
    SqliteKVStoreConfig,
    SqliteSqlStoreConfig,
    SqlStoreReference,
)
from llama_stack.core.utils.config_dirs import DISTRIBS_BASE_DIR, EXTERNAL_PROVIDERS_DIR
from llama_stack.core.utils.dynamic import instantiate_class_type
from llama_stack.core.utils.image_types import LlamaStackImageType
from llama_stack.providers.datatypes import Api

TEMPLATES_PATH = Path(__file__).parent.parent.parent / "distributions"


class ImageType(Enum):
    CONTAINER = "container"
    VENV = "venv"


def print_subcommand_description(parser, subparsers):
    """Print descriptions of subcommands."""
    description_text = ""
    for name, subcommand in subparsers.choices.items():
        description = subcommand.description
        description_text += f"  {name:<21} {description}\n"
    parser.epilog = description_text


def generate_run_config(
    build_config: BuildConfig,
    build_dir: Path,
    image_name: str,
) -> Path:
    """
    Generate a run.yaml template file for user to edit from a build.yaml file
    """
    apis = list(build_config.distribution_spec.providers.keys())
    distro_dir = DISTRIBS_BASE_DIR / image_name
    run_config = StackRunConfig(
        container_image=(image_name if build_config.image_type == LlamaStackImageType.CONTAINER.value else None),
        image_name=image_name,
        apis=apis,
        providers={},
        storage=StorageConfig(
            backends={
                "kv_default": SqliteKVStoreConfig(db_path=str(distro_dir / "kvstore.db")),
                "sql_default": SqliteSqlStoreConfig(db_path=str(distro_dir / "sql_store.db")),
            },
            stores=ServerStoresConfig(
                metadata=KVStoreReference(backend="kv_default", namespace="registry"),
                inference=InferenceStoreReference(backend="sql_default", table_name="inference_store"),
                conversations=SqlStoreReference(backend="sql_default", table_name="openai_conversations"),
            ),
        ),
        external_providers_dir=build_config.external_providers_dir
        if build_config.external_providers_dir
        else EXTERNAL_PROVIDERS_DIR,
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


@lru_cache
def available_templates_specs() -> dict[str, BuildConfig]:
    import yaml

    template_specs = {}
    for p in TEMPLATES_PATH.rglob("*build.yaml"):
        template_name = p.parent.name
        with open(p) as f:
            build_config = BuildConfig(**yaml.safe_load(f))
            template_specs[template_name] = build_config
    return template_specs
