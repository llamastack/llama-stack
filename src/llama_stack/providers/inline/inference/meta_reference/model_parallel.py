# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Callable
from functools import partial
from typing import Any

from llama_stack.models.llama.llama3.chat_format import ChatFormat as Llama3ChatFormat
from llama_stack.models.llama.llama4.chat_format import ChatFormat as Llama4ChatFormat

from .parallel_utils import ModelParallelProcessGroup


class ModelRunner:
    def __init__(self, llama):
        self.llama = llama

    def __call__(self, task: Any):
        task_type = task[0]
        if task_type == "chat_completion":
            # task[1] is [params, raw_messages]
            params, raw_messages = task[1]
            return self.llama.chat_completion(params, raw_messages)
        else:
            raise ValueError(f"Unexpected task type {task_type}")


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
