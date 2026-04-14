# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml

from llama_stack.core.datatypes import LLAMA_STACK_RUN_CONFIG_VERSION
from llama_stack.core.distribution import discover_entry_point_providers
from llama_stack.core.storage.datatypes import (
    InferenceStoreReference,
    KVStoreReference,
    SqlStoreReference,
)
from llama_stack.core.storage.kvstore.config import SqliteKVStoreConfig
from llama_stack.core.storage.sqlstore.sqlstore import SqliteSqlStoreConfig
from llama_stack.core.utils.config_resolution import discover_distribution_packages
from llama_stack.core.utils.dynamic import instantiate_class_type
from llama_stack.log import get_logger
from llama_stack_api import ProviderSpec, RemoteProviderSpec

logger = get_logger(name=__name__, category="core")


def deep_merge(base: dict, overlay: dict) -> dict:
    """Deep-merge overlay onto base.

    - Dicts are merged recursively (overlay keys take precedence).
    - All other types (lists, scalars, None) in the overlay replace
      the base value entirely.
    - Base values not present in the overlay are preserved.
    """
    merged = copy.deepcopy(base)
    for key, overlay_value in overlay.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(overlay_value, dict):
            merged[key] = deep_merge(base_value, overlay_value)
        else:
            merged[key] = copy.deepcopy(overlay_value)
    return merged


def _get_provider_config(spec: ProviderSpec, distro_dir: str) -> dict[str, Any]:
    """Call sample_run_config() on a provider's config class to get default config."""
    if not spec.config_class:
        return {}

    config_class = instantiate_class_type(spec.config_class)
    if not hasattr(config_class, "sample_run_config"):
        return {}

    try:
        result: dict[str, Any] = config_class.sample_run_config(__distro_dir__=distro_dir)
    except TypeError:
        result = config_class.sample_run_config()
    return result


def _derive_provider_id(spec: ProviderSpec) -> str:
    """Derive a provider_id from a provider spec."""
    if isinstance(spec, RemoteProviderSpec):
        return spec.adapter_type
    return spec.provider_type.split("::")[-1]


def _discover_distro_name() -> str:
    """Discover the distribution name from the installed distribution entry point.

    Expects exactly one distribution package in the current environment.
    """
    distros = discover_distribution_packages()
    if len(distros) == 1:
        return distros[0]
    elif len(distros) == 0:
        raise ValueError("Failed to discover distribution: no distribution package found in environment")
    else:
        raise ValueError(f"Failed to discover distribution: multiple distribution packages found: {', '.join(distros)}")


def _default_storage_config(distro_dir: str) -> dict[str, Any]:
    """Build the default storage config for a distribution."""
    return {
        "backends": {
            "kv_default": SqliteKVStoreConfig.sample_run_config(
                __distro_dir__=distro_dir,
                db_name="kvstore.db",
            ),
            "sql_default": SqliteSqlStoreConfig.sample_run_config(
                __distro_dir__=distro_dir,
                db_name="sql_store.db",
            ),
        },
        "stores": {
            "metadata": KVStoreReference(
                backend="kv_default",
                namespace="registry",
            ).model_dump(exclude_none=True),
            "inference": InferenceStoreReference(
                backend="sql_default",
                table_name="inference_store",
            ).model_dump(exclude_none=True),
            "conversations": SqlStoreReference(
                backend="sql_default",
                table_name="openai_conversations",
            ).model_dump(exclude_none=True),
            "prompts": KVStoreReference(
                backend="kv_default",
                namespace="prompts",
            ).model_dump(exclude_none=True),
            "connectors": KVStoreReference(
                backend="kv_default",
                namespace="connectors",
            ).model_dump(exclude_none=True),
        },
    }


def build_base_config(distro_name: str) -> dict[str, Any]:
    """Build a base config by discovering all installed providers via entry points.

    Discovers all providers registered through Python entry points, generates
    default configurations for each, and assembles a complete StackConfig with
    default storage and server settings.
    """
    providers = discover_entry_point_providers()

    if not providers:
        logger.warning("No providers discovered via entry points")

    providers_by_api: dict[str, list[dict[str, Any]]] = {}
    distro_dir = f"~/.llama/distributions/{distro_name}"

    for spec in providers:
        api_str = spec.api.value
        if api_str not in providers_by_api:
            providers_by_api[api_str] = []

        provider_id = _derive_provider_id(spec)
        config = _get_provider_config(spec, distro_dir)

        provider_entry: dict[str, Any] = {
            "provider_id": provider_id,
            "provider_type": spec.provider_type,
        }
        if config:
            provider_entry["config"] = config

        providers_by_api[api_str].append(provider_entry)

    apis = sorted(providers_by_api.keys())

    return {
        "version": LLAMA_STACK_RUN_CONFIG_VERSION,
        "distro_name": distro_name,
        "apis": apis,
        "providers": providers_by_api,
        "storage": _default_storage_config(distro_dir),
        "registered_resources": {
            "models": [],
            "shields": [],
            "vector_dbs": [],
        },
        "server": {
            "port": 8321,
        },
    }


def load_patch_chain(patch_path: Path, _seen: set[Path] | None = None) -> list[dict[str, Any]]:
    """Load a patch file and recursively resolve its _base chain.

    If the patch contains a ``_base`` key, the referenced patch file
    (resolved relative to the same directory) is loaded first. This
    continues recursively until a patch with no ``_base`` is reached.

    Returns patches in application order: base-most first, requested
    patch last.
    """
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch file not found: {patch_path}")

    resolved = patch_path.resolve()
    if _seen is None:
        _seen = set()
    if resolved in _seen:
        raise ValueError(f"Circular _base reference detected: {patch_path}")
    _seen.add(resolved)

    with open(patch_path) as f:
        patch = yaml.safe_load(f)

    if not patch:
        return []

    chain: list[dict[str, Any]] = []
    base_ref = patch.pop("_base", None)
    if base_ref:
        base_path = patch_path.parent / base_ref
        chain = load_patch_chain(base_path, _seen)

    chain.append(patch)
    return chain


def generate_config(patch_path: str | None = None) -> dict[str, Any]:
    """Generate a complete run config from the installed environment and patches.

    1. Discover the distribution name from installed entry points
    2. Build a base config from discovered entry-point providers
    3. Load the patch chain (recursively resolving _base references)
    4. Deep-merge each patch in order onto the base
    5. Return the final config dict
    """
    distro_name = _discover_distro_name()

    config = build_base_config(distro_name)

    if patch_path:
        patches = load_patch_chain(Path(patch_path))
        for patch in patches:
            config = deep_merge(config, patch)

    return config


def run_generate_config_command(args: argparse.Namespace) -> None:
    config = generate_config(patch_path=args.patch)

    output = yaml.safe_dump(config, sort_keys=False)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(output)
        print(f"Config written to {output_path}", file=sys.stderr)
    else:
        print(output)
