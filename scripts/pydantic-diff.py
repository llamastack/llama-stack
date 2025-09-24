# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.core.datatypes import StackRunConfig


def test_build_config_v1_schema_is_unchanged(snapshot):
    """
    Ensures the V1 schema never changes.
    """
    snapshot.assert_match(StackRunConfig.model_json_schema(), "stored_build_config_v1_schema.json")
