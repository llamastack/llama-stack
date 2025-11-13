# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, IAny, nc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import copy
import json
import multiprocessing
import os
import tempfile
import time
import uuid
from collections.abc import Callable, Generator
from enum import Enum
from typing import Annotated, Literal

import torch
import zmq
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_group,
    get_model_parallel_rank,
    get_model_parallel_src_rank,
)
from pydantic import BaseModel, Field
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import GenerationResult

log = get_logger(name=__name__, category="inference")


class ProcessingMessageName(str, Enum):
    ready_request = "ready_request"
    ready_response = "ready_response"
    end_sentinel = "end_sentinel"
    cancel_sentinel = "cancel_sentinel"
    task_request = "task_request"
    task_response = "task_response"
    exception_response = "exception_response"


class ReadyRequest(BaseModel):
    type: Literal[ProcessingMessageName.ready_request] = ProcessingMessageName.ready_request


class ReadyResponse(BaseModel):
    type: Literal[ProcessingMessageName.ready_response] = ProcessingMessageName.ready_response


class EndSentinel(BaseModel):
    type: Literal[ProcessingMessageName.end_sentinel] = ProcessingMessageName.end_sentinel


class CancelSentinel(BaseModel):
    type: Literal[ProcessingMessageName.cancel_sentinel] = ProcessingMessageName.cancel_sentinel


class TaskRequest(BaseModel):
    type: Literal[ProcessingMessageName.task_request] = ProcessingMessageName.task_request
    task: tuple[str, list]


class TaskResponse(BaseModel):
    type: Literal[ProcessingMessageName.task_response] = ProcessingMessageName.task_response
    result: list[GenerationResult]


class ExceptionResponse(BaseModel):
    type: Literal[ProcessingMessageName.exception_response] = ProcessingMessageName.exception_response
    error: str


ProcessingMessage = (
    ReadyRequest | ReadyResponse | EndSentinel | CancelSentinel | TaskRequest | TaskResponse | ExceptionResponse
)


class ProcessingMessageWrapper(BaseModel):
    payload: Annotated[
        ProcessingMessage,
        Field(discriminator="type"),
    ]


def mp_rank_0() -> bool:
    return bool(get_model_parallel_rank() == 0)


def encode_msg(msg: ProcessingMessage) -> bytes:
    return ProcessingMessageWrapper(payload=msg).model_dump_json().encode("utf-8")


def retrieve_requests(reply_socket_url: str):
    if mp_rank_0():
        context = zmq.Context()
        reply_socket = context.socket(zmq.ROUTER)
        reply_socket.connect(reply_socket_url)

        while True:
            client_id, obj = maybe_get_work(reply_socket)
            if obj is None:
                time.sleep(0.01)
                continue

            ready_response = ReadyResponse()
            reply_socket.send_multipart([client_id, encode_msg(ready_response)])
            break

    def send_obj(obj: ProcessingMessage):
        reply_socket.send_multipart([client_id, encode_msg(obj)])

    while True:
        tasks: list[ProcessingMessage | None] = [None]
        if mp_rank_0():
            client_id, maybe_task_json = maybe_get_work(reply_socket)
            if maybe_task_json is not None:
                task = maybe_parse_message(maybe_task_json)
                # there is still an unknown unclean GeneratorExit happening resulting in a
                # cancel sentinel getting queued _after_ we have finished sending everything :/
                # kind of a hack this is :/
                if task is not None and not isinstance(task, CancelSentinel):
                    tasks = [task]

        torch.distributed.broadcast_object_list(
            tasks,
            src=get_model_parallel_src_rank(),
            group=get_model_parallel_group(),
        )

        task = tasks[0]
        if task is None:
            time.sleep(0.1)
        else:
            try:
                out = yield task
                if out is None:
                    break

                for obj in out:
                    updates: list[ProcessingMessage | None] = [None]
                    if mp_rank_0():
                        _, update_json = maybe_get_work(reply_socket)
                        update = maybe_parse_message(update_json)
                        if isinstance(update, CancelSentinel):
                            updates = [update]
                        else:
                            # only send the update if it's not cancelled otherwise the object sits in the socket
                            # and gets pulled in the next request lol
                            send_obj(TaskResponse(result=obj))

                    torch.distributed.broadcast_object_list(
                        updates,
                        src=get_model_parallel_src_rank(),
                        group=get_model_parallel_group(),
                    )
                    if isinstance(updates[0], CancelSentinel):
                        log.info("quitting generation loop because request was cancelled")
                        break

                if mp_rank_0():
                    send_obj(EndSentinel())
            except Exception as e:
                log.exception("exception in generation loop")

                if mp_rank_0():
                    send_obj(ExceptionResponse(error=str(e)))

    if mp_rank_0():
        send_obj(EndSentinel())


def maybe_get_work(sock: zmq.Socket):
    message = None
    client_id = None
    try:
        client_id, obj = sock.recv_multipart(zmq.NOBLOCK)
        message = obj.decode("utf-8")
    except zmq.ZMQError as e:
        if e.errno != zmq.EAGAIN:
            raise e

    return client_id, message


def maybe_parse_message(maybe_json: str | None) -> ProcessingMessage | None:
    if maybe_json is None:
        return None
    try:
        return parse_message(maybe_json)
    except json.JSONDecodeError:
        return None
    except ValueError:
        return None


def parse_message(json_str: str) -> ProcessingMessage:
    data = json.loads(json_str)
    return copy.deepcopy(ProcessingMessageWrapper(**data).payload)


def worker_process_entrypoint(
    reply_socket_url: str,
    init_model_cb: Callable,
) -> None:
    model = init_model_cb()
    torch.distributed.barrier()
    time.sleep(1)

    # run the requests co-routine which retrieves requests from the socket
    # and sends responses (we provide) back to the caller
    req_gen = retrieve_requests(reply_socket_url)
    result = None
    while True:
        try:
            task = req_gen.send(result)
            if isinstance(task, EndSentinel):
                break

            assert isinstance(task, TaskRequest), task
            result = model(task.task)
        except StopIteration:
            break

    log.info("[debug] worker process done")


def launch_dist_group(
    reply_socket_url: str,
    model_parallel_size: int,
    init_model_cb: Callable,
    **kwargs,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # TODO: track workers and if they terminate, tell parent process about it so cleanup can happen
        launch_config = LaunchConfig(
            max_nodes=1,
            min_nodes=1,
            nproc_per_node=model_parallel_size,
            start_method="fork",
            rdzv_backend="c10d",
            rdzv_endpoint=os.path.join(tmpdir, "rdzv"),
            rdzv_configs={"store_type": "file", "timeout": 90},
            max_restarts=0,
            monitor_interval=1,
            run_id=str(uuid.uuid4()),
        )
        elastic_launch(launch_config, entrypoint=worker_process_entrypoint)(
            reply_socket_url,
            init_model_cb,
        )


def start_model_parallel_process(
    model_parallel_size: int,
    init_model_cb: Callable,
    **kwargs,
):
    context = zmq.Context()
    request_socket = context.socket(zmq.DEALER)

    # Binding the request socket to a random port
    request_socket.bind("tcp://127.0.0.1:0")

    main_process_url = request_socket.getsockopt_string(zmq.LAST_ENDPOINT)

    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(
        target=launch_dist_group,
        args=(
            main_process_url,
            model_parallel_size,
            init_model_cb,
        ),
        kwargs=kwargs,
    )
    process.start()

    # wait until the model is loaded; rank 0 will send a message to indicate it's ready

    request_socket.send(encode_msg(ReadyRequest()))
    _response = request_socket.recv()
    log.info("Loaded model...")

    return request_socket, process


class ModelParallelProcessGroup:
    def __init__(
        self,
        model_parallel_size: int,
        init_model_cb: Callable,
        **kwargs,
    ):
        self.model_parallel_size = model_parallel_size
        self.init_model_cb = init_model_cb
        self.started = False
        self.running = False

    def start(self):
        assert not self.started, "process group already started"
        self.request_socket, self.process = start_model_parallel_process(
            self.model_parallel_size,
            self.init_model_cb,
        )
        self.started = True

    def stop(self):
        assert self.started, "process group not started"
        if self.process.is_alive():
            self.request_socket.send(encode_msg(EndSentinel()), zmq.NOBLOCK)
            self.process.join()
        self.started = False

    def run_inference(
        self,
        req: tuple[str, list],
    ) -> Generator:
        assert not self.running, "inference already running"

        self.running = True
        try:
            self.request_socket.send(encode_msg(TaskRequest(task=req)))
            while True:
                obj_json = self.request_socket.recv()
                obj = parse_message(obj_json)

                if isinstance(obj, EndSentinel):
                    break

                if isinstance(obj, ExceptionResponse):
                    log.error(f"[debug] got exception {obj.error}")
                    raise Exception(obj.error)

                if isinstance(obj, TaskResponse):
                    yield obj.result

        except GeneratorExit:
            self.request_socket.send(encode_msg(CancelSentinel()))
            while True:
                obj_json = self.request_socket.send()
                obj = parse_message(obj_json)
                if isinstance(obj, EndSentinel):
                    break
        finally:
            self.running = False
