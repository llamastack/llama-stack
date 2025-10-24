# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Callable, Generator
from copy import deepcopy
from functools import partial
from typing import Any

from llama_stack.models.llama.llama3.chat_format import ChatFormat as Llama3ChatFormat
from llama_stack.models.llama.llama4.chat_format import ChatFormat as Llama4ChatFormat
from llama_stack.providers.utils.inference.prompt_adapter import (
    ChatCompletionRequestWithRawContent,
    CompletionRequestWithRawContent,
)

from .parallel_utils import ModelParallelProcessGroup


class ModelRunner:
    def __init__(self, llama):
        self.llama = llama

    # the `task` object is the same that is sent to `ModelParallelProcessGroup.run_inference()`
    def __call__(self, task: Any):
        if task[0] == "chat_completion":
            return self.llama.chat_completion(task[1])
        else:
            raise ValueError(f"Unexpected task type {task[0]}")


def init_model_cb(
    builder_fn: Callable,
    params: list[Any],
):
    llama = builder_fn(*params)
    return ModelRunner(llama)


class LlamaModelParallelGenerator:
    """
    This abstraction exists so
     - we can run model parallel code without needing to run the CLIs via torchrun
     - this also enables use model parallel code within a notebook context.

    A Context Manager is used to ensure that the model parallel process is started and stopped
    correctly. This does make the ergonomics a little awkward, because it isn't immediately
    clear at the callsite why we need to use a context manager.
    """

    def __init__(
        self,
        model_parallel_size: int,
        builder_fn: Callable,
        builder_params: list[Any],
        formatter: Llama3ChatFormat | Llama4ChatFormat,
    ):
        self.model_parallel_size = model_parallel_size
        self.builder_fn = builder_fn
        self.builder_params = builder_params
        self.formatter = formatter

    def start(self):
        self.__enter__()

    def stop(self):
        self.__exit__(None, None, None)

    def __enter__(self):
        self.group = ModelParallelProcessGroup(
            self.model_parallel_size,
            init_model_cb=partial(init_model_cb, self.builder_fn, self.builder_params),
        )
        self.group.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.group.stop()

    def completion(
        self,
        request_batch: list[CompletionRequestWithRawContent],
    ) -> Generator:
        req_obj = deepcopy(request_batch)
        gen = self.group.run_inference(("completion", req_obj))
        yield from gen

    def chat_completion(
        self,
        request_batch: list[ChatCompletionRequestWithRawContent],
    ) -> Generator:
        req_obj = deepcopy(request_batch)
        gen = self.group.run_inference(("chat_completion", req_obj))
        yield from gen
