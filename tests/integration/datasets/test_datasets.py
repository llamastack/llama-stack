# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import base64
import mimetypes
import os

import pytest

# How to run this test:
#
# LLAMA_STACK_CONFIG="template-name" pytest -v tests/integration/datasets


def data_url_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as file:
        file_content = file.read()

    base64_content = base64.b64encode(file_content).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)

    data_url = f"data:{mime_type};base64,{base64_content}"

    return data_url


@pytest.mark.parametrize(
    "purpose, source, provider_id, limit",
    [
        (
            "eval/messages-answer",
            {
                "type": "uri",
                "uri": "huggingface://datasets/llamastack/simpleqa?split=train",
            },
            "huggingface",
            10,
        ),
        (
            "eval/messages-answer",
            {
                "type": "rows",
                "rows": [
                    {
                        "messages": [{"role": "user", "content": "Hello, world!"}],
                        "answer": "Hello, world!",
                    },
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "What is the capital of France?",
                            }
                        ],
                        "answer": "Paris",
                    },
                ],
            },
            "localfs",
            2,
        ),
        (
            "eval/messages-answer",
            {
                "type": "uri",
                "uri": data_url_from_file(os.path.join(os.path.dirname(__file__), "test_dataset.csv")),
            },
            "localfs",
            5,
        ),
    ],
)
def test_register_and_iterrows(llama_stack_client, purpose, source, provider_id, limit):
    dataset = llama_stack_client.beta.datasets.register(
        purpose=purpose,
        source=source,
    )
    assert dataset.identifier is not None
    assert dataset.provider_id == provider_id
    iterrow_response = llama_stack_client.beta.datasets.iterrows(dataset.identifier, limit=limit)
    assert len(iterrow_response.data) == limit

    dataset_list = llama_stack_client.beta.datasets.list()
    assert dataset.identifier in [d.identifier for d in dataset_list]

    llama_stack_client.beta.datasets.unregister(dataset.identifier)
    dataset_list = llama_stack_client.beta.datasets.list()
    assert dataset.identifier not in [d.identifier for d in dataset_list]
