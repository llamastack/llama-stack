# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import copy
import importlib.metadata
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from llama_stack_api import ProviderSpec, RemoteProviderSpec
from pydantic import BaseModel

from llama_stack.core.datatypes import (
    LLAMA_STACK_RUN_CONFIG_VERSION,
    SafetyConfig,
    VectorStoresConfig,
)
from llama_stack.core.storage.datatypes import (
    InferenceStoreReference,
    KVStoreReference,
    SqlStoreReference,
)
from llama_stack.core.storage.kvstore.config import (
    PostgresKVStoreConfig,
    SqliteKVStoreConfig,
)
from llama_stack.core.storage.sqlstore.sqlstore import (
    PostgresSqlStoreConfig,
    SqliteSqlStoreConfig,
)
from llama_stack.core.utils.config_resolution import discover_distribution_packages
from llama_stack.core.utils.dynamic import instantiate_class_type
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="core")


# ---------------------------------------------------------------------------
# Overlay schema
# ---------------------------------------------------------------------------


class CompositionDirectives(BaseModel):
    """Pre-merge directives that establish a foundation before patch data is merged."""

    storage: Literal["sqlite", "postgres"] | None = None


class FinalizerDirectives(BaseModel):
    """Post-merge directives that transform the config after patch data is merged."""

    conditional_providers: dict[str, str] | None = None


class ConfigOverlay(BaseModel):
    """Schema for distribution overlay YAML files.

    Execution order per overlay in the chain:
        1. composition — pre-merge setup (e.g. swap storage to postgres)
        2. patch — strategic merge onto the config
        3. finalizers — post-merge transforms (e.g. conditional provider ID wrapping)
    """

    model_config = {"extra": "forbid"}

    base: str | None = None
    composition: CompositionDirectives | None = None
    patch: dict[str, Any] | None = None
    finalizers: FinalizerDirectives | None = None


# ---------------------------------------------------------------------------
# Merge utilities
# ---------------------------------------------------------------------------


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


def _strategic_merge_provider_lists(
    base_list: list[dict[str, Any]],
    patch_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge two provider lists using provider_type as the merge key.

    - Patch entries matching a base entry by provider_type are deep-merged
      (patch values override base values).
    - Patch entries with no base match are appended.
    - Base entries not mentioned in the patch are preserved.
    """
    base_by_type: dict[str, dict[str, Any]] = {}
    base_order: list[str] = []
    for entry in base_list:
        ptype = entry.get("provider_type", "")
        base_by_type[ptype] = entry
        base_order.append(ptype)

    patch_types_seen: set[str] = set()
    merged: list[dict[str, Any]] = []

    # First pass: emit base entries in order, merging with patch if present
    patch_by_type: dict[str, dict[str, Any]] = {}
    for entry in patch_list:
        ptype = entry.get("provider_type", "")
        patch_by_type[ptype] = entry
        patch_types_seen.add(ptype)

    for ptype in base_order:
        base_entry = base_by_type[ptype]
        if ptype in patch_by_type:
            merged.append(deep_merge(base_entry, patch_by_type[ptype]))
        else:
            merged.append(copy.deepcopy(base_entry))

    # Second pass: append patch entries not in base
    for entry in patch_list:
        ptype = entry.get("provider_type", "")
        if ptype not in base_by_type:
            merged.append(copy.deepcopy(entry))

    return merged


def strategic_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge with strategic list merging for provider lists.

    Provider lists under the ``providers`` key are merged by matching
    on ``provider_type`` (like Kubernetes strategic merge patch).
    All other keys use standard deep_merge.
    """
    base_providers = base.get("providers", {})
    patch_providers = patch.pop("providers", None)

    merged = deep_merge(base, patch)

    if patch_providers is not None:
        merged_providers = copy.deepcopy(base_providers)
        for api_str, patch_list in patch_providers.items():
            base_list = base_providers.get(api_str, [])
            merged_providers[api_str] = _strategic_merge_provider_lists(base_list, patch_list)
        merged["providers"] = merged_providers

    return merged


# ---------------------------------------------------------------------------
# Provider discovery
# ---------------------------------------------------------------------------


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


def _get_distribution_provider_packages(distribution_name: str) -> set[str]:
    """Get the set of provider package names declared as direct dependencies of a distribution.

    This is useful in development environments (e.g. uv workspaces) where all
    providers are installed transitively through the core package, but we only
    want to generate configs for the providers the distribution explicitly
    depends on.
    """
    try:
        dist = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        logger.warning("Failed to find distribution package", distribution=distribution_name)
        return set()

    packages: set[str] = set()
    for req_str in dist.requires or []:
        name = req_str.split(";")[0].split("[")[0].split(">")[0].split("<")[0].split("=")[0].split("!")[0].strip()
        if name.startswith("llama-stack-provider-"):
            packages.add(name)
    return packages


def _discover_providers_for_distribution(distribution_name: str | None) -> list[ProviderSpec]:
    """Discover providers, optionally filtered by a distribution's declared dependencies."""
    eps = importlib.metadata.entry_points(group="llama_stack.providers")

    allowed_packages: set[str] | None = None
    if distribution_name:
        allowed_packages = _get_distribution_provider_packages(distribution_name)
        if not allowed_packages:
            logger.warning(
                "No provider packages found in distribution dependencies",
                distribution=distribution_name,
            )

    seen_per_api: dict[str, set[str]] = {}
    results: list[ProviderSpec] = []
    for ep in eps:
        ep_dist_name = ep.dist.name if ep.dist else None
        if allowed_packages is not None and ep_dist_name not in allowed_packages:
            continue
        try:
            get_spec = ep.load()
            spec_or_specs = get_spec()
            specs = spec_or_specs if isinstance(spec_or_specs, list) else [spec_or_specs]
            for spec in specs:
                if getattr(spec, "deprecation_warning", None):
                    continue
                api_key = spec.api.value
                if api_key not in seen_per_api:
                    seen_per_api[api_key] = set()
                if spec.provider_type in seen_per_api[api_key]:
                    continue
                seen_per_api[api_key].add(spec.provider_type)
                results.append(spec)
        except Exception as e:
            logger.warning(
                "Failed to load provider from entry point",
                entry_point=ep.name,
                error=str(e),
            )
    return results


# ---------------------------------------------------------------------------
# Storage config builders
# ---------------------------------------------------------------------------


def _sqlite_storage_config(distro_dir: str) -> dict[str, Any]:
    """Build SQLite storage config for a distribution."""
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
        "stores": _default_stores(),
    }


def _postgres_storage_config() -> dict[str, Any]:
    """Build PostgreSQL storage config for a distribution."""
    return {
        "backends": {
            "kv_default": PostgresKVStoreConfig.sample_run_config(),
            "sql_default": PostgresSqlStoreConfig.sample_run_config(),
        },
        "stores": _default_stores(),
    }


def _default_stores() -> dict[str, Any]:
    """Build the default store references shared by all storage backends."""
    return {
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
    }


# ---------------------------------------------------------------------------
# Overlay directive processing
# ---------------------------------------------------------------------------


def _apply_composition(config: dict[str, Any], composition: CompositionDirectives) -> dict[str, Any]:
    """Apply composition directives that set up state BEFORE the patch merge.

    These directives establish a foundation that patches can then override.
    """
    config = copy.deepcopy(config)

    if composition.storage == "postgres":
        config["storage"] = _postgres_storage_config()

    return config


def _apply_finalizers(config: dict[str, Any], finalizers: FinalizerDirectives) -> dict[str, Any]:
    """Apply finalizer directives that transform config AFTER the patch merge.

    These directives transform values that may have been introduced by patches.
    """
    config = copy.deepcopy(config)

    if finalizers.conditional_providers:
        for _api_str, provider_list in config.get("providers", {}).items():
            for entry in provider_list:
                pid = entry.get("provider_id", "")
                if pid in finalizers.conditional_providers:
                    env_var = finalizers.conditional_providers[pid]
                    entry["provider_id"] = f"${{env.{env_var}:+{pid}}}"

    return config


# ---------------------------------------------------------------------------
# Base config builder
# ---------------------------------------------------------------------------


def build_base_config(distro_name: str, distribution_package: str | None = None) -> dict[str, Any]:
    """Build a base config by discovering installed providers via entry points.

    Discovers providers registered through Python entry points, generates
    default configurations for each, and assembles a complete config with
    default storage, server, vector_stores, safety, and resource settings.

    Args:
        distro_name: Name used for the distro_dir path and config metadata.
        distribution_package: If provided, only include providers that are
            direct dependencies of this distribution package. Useful in
            development environments where all providers are installed.
    """
    providers = _discover_providers_for_distribution(distribution_package)

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

    # Sort providers within each API by provider_type for deterministic output
    for api_str in providers_by_api:
        providers_by_api[api_str].sort(key=lambda p: p.get("provider_type", ""))

    apis = sorted(providers_by_api.keys())

    return {
        "version": LLAMA_STACK_RUN_CONFIG_VERSION,
        "distro_name": distro_name,
        "apis": apis,
        "providers": providers_by_api,
        "storage": _sqlite_storage_config(distro_dir),
        "registered_resources": {
            "models": [],
            "shields": [],
            "vector_dbs": [],
        },
        "server": {
            "port": 8321,
        },
        "vector_stores": VectorStoresConfig().model_dump(exclude_none=True),
        "safety": SafetyConfig().model_dump(exclude_none=True),
        "connectors": [],
    }


# ---------------------------------------------------------------------------
# Overlay chain loader
# ---------------------------------------------------------------------------


def load_overlay_chain(overlay_path: Path, _seen: set[Path] | None = None) -> list[ConfigOverlay]:
    """Load an overlay file and recursively resolve its ``base`` chain.

    If the overlay contains a ``base`` key, the referenced overlay file
    (resolved relative to the same directory) is loaded first. This
    continues recursively until an overlay with no ``base`` is reached.

    Returns ConfigOverlay objects in application order: base-most first,
    requested overlay last.
    """
    if not overlay_path.exists():
        raise FileNotFoundError(f"Overlay file not found: {overlay_path}")

    resolved = overlay_path.resolve()
    if _seen is None:
        _seen = set()
    if resolved in _seen:
        raise ValueError(f"Circular base reference detected: {overlay_path}")
    _seen.add(resolved)

    with open(overlay_path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        return []

    overlay = ConfigOverlay(**raw)

    chain: list[ConfigOverlay] = []
    if overlay.base:
        base_path = overlay_path.parent / overlay.base
        chain = load_overlay_chain(base_path, _seen)

    chain.append(overlay)
    return chain


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def generate_config(overlay_path: str | None = None, distribution: str | None = None) -> dict[str, Any]:
    """Generate a complete run config from the installed environment and overlays.

    1. Discover the distribution name (from --distribution or entry points)
    2. Build a complete base config from discovered entry-point providers
    3. Load the overlay chain (recursively resolving ``base`` references)
    4. For each overlay in the chain:
       a. Apply ``composition`` directives (pre-merge setup, e.g. storage)
       b. Strategic-merge patch data onto config
       c. Apply ``finalizers`` directives (post-merge transforms, e.g. conditional_providers)
    5. Return the final config dict

    Patch data uses strategic merge: provider lists are merged by matching on
    ``provider_type`` (like Kubernetes strategic merge patch), while all
    other keys use standard recursive deep merge.

    Args:
        overlay_path: Optional path to an overlay YAML file.
        distribution: Optional distribution package name to filter providers by.
            If provided, only providers that are direct dependencies of this
            distribution will be included in the base config.
    """
    if distribution:
        prefix = "llama-stack-distribution-"
        if distribution.startswith(prefix):
            distro_name = distribution[len(prefix) :]
        else:
            distro_name = distribution
    else:
        distro_name = _discover_distro_name()

    config = build_base_config(distro_name, distribution_package=distribution)

    if overlay_path:
        overlays = load_overlay_chain(Path(overlay_path))
        for overlay in overlays:
            if overlay.composition:
                config = _apply_composition(config, overlay.composition)
            if overlay.patch:
                config = strategic_merge(config, overlay.patch)
            if overlay.finalizers:
                config = _apply_finalizers(config, overlay.finalizers)

    return config


def run_generate_config_command(args: argparse.Namespace) -> None:
    config = generate_config(
        overlay_path=args.overlay,
        distribution=getattr(args, "distribution", None),
    )

    output = yaml.safe_dump(config, sort_keys=False)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(output)
        print(f"Config written to {output_path}", file=sys.stderr)
    else:
        print(output)
