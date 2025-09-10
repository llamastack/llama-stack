# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import yaml

from llama_stack.apis.datatypes import Api, ExternalApiSpec
from llama_stack.core.datatypes import BuildConfig, StackRunConfig
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="core")


def load_external_apis(config: StackRunConfig | BuildConfig | None) -> dict[Api, ExternalApiSpec]:
    """Load external API specifications from the configured directory.

    Args:
        config: StackRunConfig or BuildConfig containing the external APIs directory path

    Returns:
        A dictionary mapping API names to their specifications
    """
    if not config or not config.external_apis_dir:
        return {}

    external_apis_dir = config.external_apis_dir.expanduser().resolve()
    if not external_apis_dir.is_dir():
        logger.error(f"External APIs directory is not a directory: {external_apis_dir}")
        return {}

    logger.info(f"Loading external APIs from {external_apis_dir}")
    external_apis: dict[Api, ExternalApiSpec] = {}

    # Look for YAML files in the external APIs directory
    for yaml_path in external_apis_dir.glob("*.yaml"):
        try:
            with open(yaml_path) as f:
                spec_data = yaml.safe_load(f)

            spec = ExternalApiSpec(**spec_data)
            api = Api.add(spec.name)
            logger.info(f"Loaded external API spec for {spec.name} from {yaml_path}")
            external_apis[api] = spec
        except yaml.YAMLError as yaml_err:
            logger.error(f"Failed to parse YAML file {yaml_path}: {yaml_err}")
            raise
        except Exception:
            logger.exception(f"Failed to load external API spec from {yaml_path}")
            raise

    return external_apis
