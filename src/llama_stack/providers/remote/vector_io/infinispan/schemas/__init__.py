# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Protobuf schemas for Infinispan vector store.
"""

import importlib.resources as pkg_resources
from pathlib import Path


def get_schema_path(schema_name: str) -> Path:
    """Get the path to a schema file."""
    try:
        # Python 3.9+
        traversable = pkg_resources.files(__name__) / schema_name
        # Convert Traversable to Path for type consistency
        return Path(str(traversable))
    except AttributeError:
        # Python 3.8 fallback
        with pkg_resources.path(__name__, schema_name) as p:
            return p


def load_schema(schema_name: str) -> str:
    """Load a schema file as a string."""
    schema_path = get_schema_path(schema_name)
    return schema_path.read_text()
