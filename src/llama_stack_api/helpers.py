# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


def remove_null_from_anyof(schema: dict) -> None:
    """Remove null type from anyOf if present in JSON schema.

    This is used to make optional fields non-nullable in the OpenAPI spec,
    matching OpenAI's API specification where optional fields can be omitted
    but cannot be explicitly set to null.

    Handles both OpenAPI 3.0 format (anyOf with null) and OpenAPI 3.1 format
    (type as array with null).
    """
    # Handle anyOf format: anyOf: [{type: integer}, {type: null}]
    if "anyOf" in schema:
        schema["anyOf"] = [s for s in schema["anyOf"] if s.get("type") != "null"]
        if len(schema["anyOf"]) == 1:
            # If only one type left, flatten it
            only_schema = schema["anyOf"][0]
            schema.pop("anyOf")
            schema.update(only_schema)

    # Handle OpenAPI 3.1 format: type: ['integer', 'null']
    elif isinstance(schema.get("type"), list) and "null" in schema["type"]:
        schema["type"].remove("null")
        if len(schema["type"]) == 1:
            schema["type"] = schema["type"][0]


__all__ = [
    "remove_null_from_anyof",
]
