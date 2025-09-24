# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

from llama_stack.core.datatypes import StackRunConfig


def test_run_config_v1_schema_is_unchanged(snapshot):
    """
    Ensures the V1 schema never changes.
    """
    schema = StackRunConfig.model_json_schema()
    snapshot.assert_match(json.dumps(schema, indent=2), "stored_run_config_v1_schema.json")
