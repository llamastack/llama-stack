# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime

import pytest
import yaml

from llama_stack.core.configure import (
    LLAMA_STACK_RUN_CONFIG_VERSION,
    parse_and_maybe_upgrade_config,
)


@pytest.fixture
def config_with_image_name_int():
    return yaml.safe_load(
        f"""
        version: {LLAMA_STACK_RUN_CONFIG_VERSION}
        image_name: 1234
        apis_to_serve: []
        built_at: {datetime.now().isoformat()}
        providers:
          inference:
            - provider_id: provider1
              provider_type: inline::meta-reference
              config: {{}}
          safety:
            - provider_id: provider1
              provider_type: inline::meta-reference
              config:
                llama_guard_shield:
                  model: Llama-Guard-3-1B
                  excluded_categories: []
                  disable_input_check: false
                  disable_output_check: false
                enable_prompt_guard: false
          memory:
            - provider_id: provider1
              provider_type: inline::meta-reference
              config: {{}}
    """
    )


@pytest.fixture
def up_to_date_config():
    return yaml.safe_load(
        f"""
        version: {LLAMA_STACK_RUN_CONFIG_VERSION}
        image_name: foo
        apis_to_serve: []
        built_at: {datetime.now().isoformat()}
        providers:
          inference:
            - provider_id: provider1
              provider_type: inline::meta-reference
              config: {{}}
          safety:
            - provider_id: provider1
              provider_type: inline::meta-reference
              config:
                llama_guard_shield:
                  model: Llama-Guard-3-1B
                  excluded_categories: []
                  disable_input_check: false
                  disable_output_check: false
                enable_prompt_guard: false
          memory:
            - provider_id: provider1
              provider_type: inline::meta-reference
              config: {{}}
    """
    )


@pytest.fixture
def old_config():
    return yaml.safe_load(
        f"""
        image_name: foo
        built_at: {datetime.now().isoformat()}
        apis_to_serve: []
        routing_table:
          inference:
            - provider_type: remote::ollama
              config:
                host: localhost
                port: 11434
              routing_key: Llama3.2-1B-Instruct
            - provider_type: inline::meta-reference
              config:
                model: Llama3.1-8B-Instruct
              routing_key: Llama3.1-8B-Instruct
          safety:
            - routing_key: ["shield1", "shield2"]
              provider_type: inline::meta-reference
              config:
                llama_guard_shield:
                  model: Llama-Guard-3-1B
                  excluded_categories: []
                  disable_input_check: false
                  disable_output_check: false
                enable_prompt_guard: false
          memory:
            - routing_key: vector
              provider_type: inline::meta-reference
              config: {{}}
        api_providers:
          telemetry:
            provider_type: noop
            config: {{}}
    """
    )


@pytest.fixture
def invalid_config():
    return yaml.safe_load(
        """
        routing_table: {}
        api_providers: {}
    """
    )


def test_parse_and_maybe_upgrade_config_up_to_date(up_to_date_config):
    result = parse_and_maybe_upgrade_config(up_to_date_config)
    assert result.version == LLAMA_STACK_RUN_CONFIG_VERSION
    assert "inference" in result.providers


def test_parse_and_maybe_upgrade_config_old_format(old_config):
    result = parse_and_maybe_upgrade_config(old_config)
    assert result.version == LLAMA_STACK_RUN_CONFIG_VERSION
    assert all(api in result.providers for api in ["inference", "safety", "memory", "telemetry"])
    safety_provider = result.providers["safety"][0]
    assert safety_provider.provider_type == "inline::meta-reference"
    assert "llama_guard_shield" in safety_provider.config

    inference_providers = result.providers["inference"]
    assert len(inference_providers) == 2
    assert {x.provider_id for x in inference_providers} == {
        "remote::ollama-00",
        "inline::meta-reference-01",
    }

    ollama = inference_providers[0]
    assert ollama.provider_type == "remote::ollama"
    assert ollama.config["port"] == 11434


def test_parse_and_maybe_upgrade_config_invalid(invalid_config):
    with pytest.raises(KeyError):
        parse_and_maybe_upgrade_config(invalid_config)


def test_parse_and_maybe_upgrade_config_image_name_int(config_with_image_name_int):
    result = parse_and_maybe_upgrade_config(config_with_image_name_int)
    assert isinstance(result.image_name, str)
