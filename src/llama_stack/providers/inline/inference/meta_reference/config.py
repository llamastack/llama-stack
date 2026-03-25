# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, field_validator

from llama_stack.models.llama.sku_list import all_registered_models
from llama_stack.models.llama.sku_types import (
    CheckpointQuantizationFormat,
    CoreModelId,
    Model,
    ModelFamily,
)
from llama_stack_api import QuantizationConfig


def _is_supported_safety_model(model: Model) -> bool:
    if model.quantization_format != CheckpointQuantizationFormat.bf16:
        return False
    return model.core_model_id in [
        CoreModelId.llama_guard_3_8b,
        CoreModelId.llama_guard_3_1b,
        CoreModelId.llama_guard_3_11b_vision,
    ]


def supported_inference_models() -> list[Model]:
    return [
        m
        for m in all_registered_models()
        if (
            m.model_family in {ModelFamily.llama3_1, ModelFamily.llama3_2, ModelFamily.llama3_3, ModelFamily.llama4}
            or _is_supported_safety_model(m)
        )
    ]


class MetaReferenceInferenceConfig(BaseModel):
    # this is a placeholder to indicate inference model id
    # the actual inference model id is dtermined by the moddel id in the request
    # Note: you need to register the model before using it for inference
    # models in the resouce list in the config.yaml config will be registered automatically
    model: str | None = None
    torch_seed: int | None = None
    max_seq_len: int = 4096
    max_batch_size: int = 1
    model_parallel_size: int | None = None

    # when this is False, we assume that the distributed process group is setup by someone
    # outside of this code (e.g., when run inside `torchrun`). that is useful for clients
    # (including our testing code) who might be using llama-stack as a library.
    create_distributed_process_group: bool = True

    # By default, the implementation will look at ~/.llama/checkpoints/<model> but you
    # can override by specifying the directory explicitly
    checkpoint_dir: str | None = None

    quantization: QuantizationConfig | None = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, model: str) -> str:
        permitted_models = supported_inference_models()
        descriptors = [m.descriptor() for m in permitted_models]
        repos = [m.huggingface_repo for m in permitted_models if m.huggingface_repo is not None]
        if model not in (descriptors + repos):
            model_list = "\n\t".join(repos)
            raise ValueError(f"Unknown model: `{model}`. Choose from [\n\t{model_list}\n]")
        return model

    @classmethod
    def sample_run_config(
        cls,
        model: str = "Llama3.2-3B-Instruct",
        checkpoint_dir: str = "${env.CHECKPOINT_DIR:=null}",
        quantization_type: str = "${env.QUANTIZATION_TYPE:=bf16}",
        model_parallel_size: str = "${env.MODEL_PARALLEL_SIZE:=0}",
        max_batch_size: str = "${env.MAX_BATCH_SIZE:=1}",
        max_seq_len: str = "${env.MAX_SEQ_LEN:=4096}",
        **kwargs,
    ) -> dict[str, Any]:
        return {
            "model": model,
            "checkpoint_dir": checkpoint_dir,
            "quantization": {
                "type": quantization_type,
            },
            "model_parallel_size": model_parallel_size,
            "max_batch_size": max_batch_size,
            "max_seq_len": max_seq_len,
        }
