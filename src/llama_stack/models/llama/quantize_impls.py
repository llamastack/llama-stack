# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# type: ignore
import collections

from llama_stack.log import get_logger

log = get_logger(name=__name__, category="models::llama")

try:
    import fbgemm_gpu.experimental.gen_ai  # noqa: F401

    log.info("Using efficient FP8 or INT4 operators in FBGEMM.")
except ImportError:
    log.error("No efficient FP8 or INT4 operators. Please install FBGEMM.")
    raise

import torch
from torch import Tensor, nn


class Fp8ScaledWeights:
    # TODO: Ugly trick so torch allows us to replace parameters
    # with our custom Fp8Weights instance. Do this properly.
    @property
    def __class__(self) -> type[nn.parameter.Parameter]:
        return nn.Parameter

    @property
    def grad_fn(self) -> None:
        return None


# pyre-fixme[4]: Attribute annotation cannot be `Any`.
# pyre-fixme[2]: Parameter annotation cannot be `Any`.
class Fp8RowwiseWeights(
    Fp8ScaledWeights,
    collections.namedtuple(
        "Fp8RowwiseWeights",
        ["weight", "scale", "shape", "activation_scale_ub"],
    ),
):
    pass


class Int4ScaledWeights:
    # TODO: Ugly trick so torch allows us to replace parameters
    # with our custom Int4Weights instance. Do this properly.
    @property
    def __class__(self) -> type[nn.parameter.Parameter]:
        return nn.Parameter

    @property
    def grad_fn(self) -> None:
        return None


# pyre-fixme[4]: Attribute annotation cannot be `Any`.
# pyre-fixme[2]: Parameter annotation cannot be `Any`.
class Int4Weights(
    Int4ScaledWeights,
    collections.namedtuple(
        "Int4Weights",
        ["weight", "scale", "zero_point", "shape"],
    ),
):
    pass


def int4_row_quantize(
    x: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_bit = 4  # Number of target bits.
    to_quant = x.reshape(-1, group_size).to(torch.float)

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-6) / max_int

    zeros = min_val + scales * (2 ** (n_bit - 1))

    out = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int)

    # Recenter output and move to int8.
    out = (out - 2 ** (n_bit - 1)).to(dtype=torch.int8).reshape(x.shape)

    # Cutlass expects column major layout for scale and zero point,
    # so we transpose here and make them contiguous.
    scales = scales.view(x.shape[0], -1).t().contiguous()
    zeros = zeros.view(x.shape[0], -1).t().contiguous()

    return out, scales, zeros


def pack_int4(x: torch.Tensor) -> torch.Tensor:
    # Given int8 x, pack adjacent int4 values into a single int8.
    low_x = x[:, ::2]
    high_x = x[:, 1::2]

    # High bits need to left shift, this also masks off extra bits.
    high_x = torch.bitwise_left_shift(high_x, 4)
    # Low bits need to have sign bits removed.
    low_x = torch.bitwise_and(low_x, 0xF)

    # Recombine into a single value with bitwise or.
    return torch.bitwise_or(low_x, high_x).contiguous()


def bmm_nt(
    x: Tensor,
    w: Fp8RowwiseWeights | Int4Weights,
    num_tokens: Tensor | None = None,
) -> Tensor:
    if isinstance(w, Fp8ScaledWeights):
        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x, num_tokens, w.activation_scale_ub)
        return torch.ops.fbgemm.f8f8bf16_rowwise_batched(xq, w.weight, x_scale, w.scale)
    elif isinstance(w, Int4ScaledWeights):
        return torch.ops.fbgemm.bf16i4bf16_rowwise_batched(x, w.weight, w.scale, w.zero_point)
    else:
        raise ValueError("Unsupported quantization type")


def ffn_swiglu(
    x: Tensor,
    w1: Fp8RowwiseWeights | Int4Weights,
    w3: Fp8RowwiseWeights | Int4Weights,
    w2: Fp8RowwiseWeights | Int4Weights,
    num_tokens: Tensor | None = None,
    is_memory_bounded: bool = False,
) -> Tensor:
    if (isinstance(w1, Fp8ScaledWeights) and isinstance(w3, Fp8ScaledWeights) and isinstance(w2, Fp8ScaledWeights)) or (
        isinstance(w1, Int4ScaledWeights) and isinstance(w3, Int4ScaledWeights) and isinstance(w2, Int4ScaledWeights)
    ):
        return ffn_swiglu_dynamic(x, w1, w3, w2, w1.activation_scale_ub, num_tokens, is_memory_bounded)

    (B, T, D) = x.shape  # noqa: N806
    (HD_L, D_) = w1.shape  # noqa: N806
    assert D_ == D

    assert isinstance(w1, Tensor)
    assert isinstance(w3, Tensor)
    x1 = x.view(B * T, D) @ w1.T
    x2 = x.view(B * T, D) @ w3.T
    z = torch.nn.functional.silu(x1) * x2
    del x1, x2
    assert isinstance(w2, Tensor)
    return (z @ w2.T).view(B, T, D)


@torch.inference_mode()
def quantize_fp8(
    w: Tensor,
    fp8_activation_scale_ub: float,
    output_device: torch.device | None = None,
) -> Fp8RowwiseWeights:
    """Quantize [n, k] weight tensor.

    Args:
        w (Tensor): [n, k] input high precision tensor to quantize.
        fp8_activation_scale_ub (float): Upper bound for activation max.
    """
    activation_scale_ub = torch.tensor(
        [fp8_activation_scale_ub],
        dtype=torch.float,
        device=output_device,
    )
    wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
    del w
    return Fp8RowwiseWeights(
        weight=wq,
        scale=w_scale,
        shape=wq.shape,
        activation_scale_ub=activation_scale_ub,
    )


@torch.inference_mode()
def quantize_int4(
    w: Tensor,
    output_device: torch.device | None = None,
) -> Int4Weights:
    """Quantize [n, k/2] weight tensor.

    Args:
        w (Tensor): [n, k/2] input high precision tensor to quantize.
    """
    if w.ndim >= 3:
        wq, scale, zero_point = zip(*[int4_row_quantize(i) for i in w], strict=False)
        wq = torch.stack([pack_int4(i) for i in wq], dim=0)
        scale = torch.stack(scale, dim=0)
        zero_point = torch.stack(zero_point, dim=0)
    else:
        wq, scale, zero_point = int4_row_quantize(w)
        wq = pack_int4(wq)
    del w
    return Int4Weights(
        weight=wq.to(output_device),
        scale=scale.to(output_device),
        zero_point=zero_point.to(output_device),
        shape=wq.shape,
    )


@torch.inference_mode()
def load_fp8(
    w: Tensor,
    w_scale: Tensor,
    fp8_activation_scale_ub: float,
    output_device: torch.device | None = None,
) -> Fp8RowwiseWeights:
    """Load FP8 [n, k] weight tensor.

    Args:
        w (Tensor): [n, k] input FP8.
        fp8_activation_scale_ub (float): Upper bound for activation max.
    """
    activation_scale_ub = torch.tensor(
        [fp8_activation_scale_ub],
        dtype=torch.float,
        device=output_device,
    )
    return Fp8RowwiseWeights(
        weight=w.to(torch.float8_e4m3fn).to(device=output_device),
        scale=w_scale.to(device=output_device),
        shape=w.shape,
        activation_scale_ub=activation_scale_ub,
    )


@torch.inference_mode()
def load_int4(
    w: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    output_device: torch.device | None = None,
) -> Int4Weights:
    """Load INT4 [n, k/2] weight tensor.

    Args:
        w (Tensor): [n, k/2] input INT4.
    """
    return Int4Weights(
        weight=w.to(torch.int8).to(device=output_device),
        scale=scale.to(device=output_device),
        zero_point=zero_point.to(device=output_device),
        shape=w.shape,
    )


def fc_dynamic(
    x: Tensor,
    w: Fp8RowwiseWeights | Int4Weights,
    activation_scale_ub: Tensor | None = None,
    num_tokens: Tensor | None = None,
    is_memory_bounded: bool = False,
) -> Tensor:
    """
    Single w8a8 fc layer with dynamic row-wise scaling, or w4a16 fc layer with dyanmic row-wise scaling
    """
    if isinstance(w, Int4Weights):
        y = torch.ops.fbgemm.bf16i4bf16_rowwise(x, w.weight, w.scale, w.zero_point)
    else:
        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x, num_tokens, activation_scale_ub)
        y = torch.ops.fbgemm.f8f8bf16_rowwise(xq, w.weight, x_scale, w.scale, use_fast_accum=True)
        del xq
    return y


def ffn_swiglu_dynamic(
    x: Tensor,
    w1: Fp8RowwiseWeights | Int4Weights,
    w3: Fp8RowwiseWeights | Int4Weights,
    w2: Fp8RowwiseWeights | Int4Weights,
    activation_scale_ub: Tensor | None = None,
    num_tokens: Tensor | None = None,
    is_memory_bounded: bool = False,
) -> Tensor:
    assert x.dim() == 3 or x.dim() == 2
    if x.dim() == 3:
        (B, T, D) = x.shape  # noqa: N806
    else:
        (T, D) = x.shape  # noqa: N806
        B = 1  # noqa: N806

    HD_L = w1.shape[0]  # noqa: N806
    assert HD_L == w3.shape[0]
    x1 = fc_dynamic(
        x.view(B * T, D),
        w1,
        activation_scale_ub,
        num_tokens,
        is_memory_bounded,
    )
    x2 = fc_dynamic(
        x.view(B * T, D),
        w3,
        activation_scale_ub,
        num_tokens,
        is_memory_bounded,
    )
    z = torch.nn.functional.silu(x1) * x2
    del x1, x2

    z_ = fc_dynamic(z, w2, activation_scale_ub, num_tokens, is_memory_bounded)

    if x.dim() == 3:
        return z_.view(B, T, D)
    else:
        return z_
