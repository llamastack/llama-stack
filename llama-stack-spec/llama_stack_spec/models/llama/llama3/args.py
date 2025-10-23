# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum


class QuantizationScheme(Enum):
    int4_weight_int8_dynamic_activation = "int4_weight_int8_dynamic_activation"


@dataclass
class QuantizationArgs:
    scheme: QuantizationScheme | None = None
    group_size: int | None = None
    spinquant: bool = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k == "scheme":
                setattr(self, k, QuantizationScheme(v))
            else:
                if hasattr(self, k):
                    setattr(self, k, v)


@dataclass
class LoRAArgs:
    rank: int
    scale: float


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False

    max_batch_size: int = 32
    max_seq_len: int = 2048

    # vision model params
    vision_chunk_size: int = -1  # image resolution for image models
    vision_max_num_chunks: int = 4
    vision_num_cross_attention_layers: int = -1

    quantization_args: QuantizationArgs | None = None
    lora_args: LoRAArgs | None = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k == "lora_args":
                setattr(self, k, LoRAArgs(**v))
            elif k == "quantization_args":
                setattr(self, k, QuantizationArgs(**v))
            else:
                if hasattr(self, k):
                    setattr(self, k, v)

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0
