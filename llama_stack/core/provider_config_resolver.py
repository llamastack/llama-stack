# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any


def resolve_provider_kvstore_references(
    providers: dict[str, list[dict[str, Any]]],
    persistence_backends: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """
    Resolve backend references in provider kvstore configs to actual backend configs.

    This allows providers to reference centralized persistence backends instead of
    duplicating kvstore configs.

    Example:
        # Provider config with backend reference
        kvstore:
          backend: kvstore
          namespace: faiss

        # Gets resolved to actual backend config
        kvstore:
          type: sqlite
          db_path: /path/to/kvstore.db
          namespace: faiss
    """
    for api, provider_list in providers.items():
        for provider in provider_list:
            config = provider.get("config", {})
            _resolve_kvstore_in_dict(config, persistence_backends)

    return providers


def _resolve_kvstore_in_dict(config: dict[str, Any], backends: dict[str, Any]) -> None:
    """Recursively find and resolve backend references in config dict."""
    for key, value in list(config.items()):
        if isinstance(value, dict):
            # Check if this dict is a backend reference
            if "backend" in value:
                backend_name = value["backend"]
                namespace = value.get("namespace")

                if backend_name not in backends:
                    raise ValueError(
                        f"Provider references backend '{backend_name}' which is not defined in persistence.backends"
                    )

                # Clone the backend config and apply namespace
                backend_config = backends[backend_name]
                resolved_config = backend_config.model_dump() if hasattr(backend_config, "model_dump") else dict(backend_config)

                if namespace:
                    resolved_config["namespace"] = namespace

                config[key] = resolved_config
            else:
                # Not a backend reference - recursively process
                _resolve_kvstore_in_dict(value, backends)
