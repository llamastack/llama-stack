# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.remote.inference.bedrock.bedrock import (
    _get_region_prefix,
    _to_inference_profile_id,
)


def test_region_prefixes():
    assert _get_region_prefix("us-east-1") == "us."
    assert _get_region_prefix("eu-west-1") == "eu."
    assert _get_region_prefix("ap-south-1") == "ap."
    assert _get_region_prefix("ca-central-1") == "us."

    # Test case insensitive
    assert _get_region_prefix("US-EAST-1") == "us."
    assert _get_region_prefix("EU-WEST-1") == "eu."
    assert _get_region_prefix("Ap-South-1") == "ap."

    # Test None region
    assert _get_region_prefix(None) == "us."


def test_model_id_conversion():
    # Basic conversion
    assert (
        _to_inference_profile_id("meta.llama3-1-70b-instruct-v1:0", "us-east-1") == "us.meta.llama3-1-70b-instruct-v1:0"
    )

    # Already has prefix
    assert (
        _to_inference_profile_id("us.meta.llama3-1-70b-instruct-v1:0", "us-east-1")
        == "us.meta.llama3-1-70b-instruct-v1:0"
    )

    # ARN should be returned unchanged
    arn = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.meta.llama3-1-70b-instruct-v1:0"
    assert _to_inference_profile_id(arn, "us-east-1") == arn

    # ARN should be returned unchanged even without region
    assert _to_inference_profile_id(arn) == arn

    # Optional region parameter defaults to us-east-1
    assert _to_inference_profile_id("meta.llama3-1-70b-instruct-v1:0") == "us.meta.llama3-1-70b-instruct-v1:0"

    # Different regions work with optional parameter
    assert (
        _to_inference_profile_id("meta.llama3-1-70b-instruct-v1:0", "eu-west-1") == "eu.meta.llama3-1-70b-instruct-v1:0"
    )
