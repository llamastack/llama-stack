# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from collections.abc import Callable

import torch
from fairscale.nn.model_parallel.initialize import get_model_parallel_rank
from torch import Tensor, nn
from torch.nn import functional as F

from llama_stack.log import get_logger

from ...datatypes import QuantizationMode
from ..model import Transformer, TransformerBlock
from ..moe import MoE

log = get_logger(name=__name__, category="models::llama")


def swiglu_wrapper_no_reduce(
    self,
    x: Tensor,
):
    from ...quantize_impls import ffn_swiglu

    return ffn_swiglu(x, self.w1.weight, self.w3.weight, self.w2.weight)


def experts_batched_swiglu_wrapper(
    self,
    x: Tensor,  # (e, g, D)
    w1: Tensor,  # (e, D, F)
    w3: Tensor,  # (e, D, F)
    w2: Tensor,  # (e, F, D)
) -> torch.Tensor:
    from ...quantize_impls import bmm_nt

    middle_out_egF = F.silu(bmm_nt(x, w1)) * bmm_nt(x, w3)  # noqa: N806
    return bmm_nt(middle_out_egF, w2)


def convert_to_quantized_model(
    model: Transformer,
    checkpoint_dir: str,
    quantization_mode: str | None = None,
    fp8_activation_scale_ub: float | None = 1200.0,
    use_rich_progress: bool = True,
) -> Transformer:
    from ...quantize_impls import (
        Fp8ScaledWeights,
        Int4ScaledWeights,
        load_fp8,
        load_int4,
        quantize_fp8,
        quantize_int4,
    )

    rank = get_model_parallel_rank()

    def should_quantize_block(block: nn.Module) -> bool:
        if not isinstance(block, TransformerBlock):
            return False

        is_moe = isinstance(block.feed_forward, MoE)
        if quantization_mode == QuantizationMode.fp8_mixed:
            # skip quantization on first and last layers
            return is_moe and not (block.layer_id == 0 or block.layer_id == (model.n_layers - 1))

        return is_moe

    use_rich_progress = use_rich_progress and rank == 0
    progress, log_status, update_status = logging_callbacks(use_rich_progress, rank, model, should_quantize_block)
    if quantization_mode == QuantizationMode.int4_mixed:
        int4_scales_path = os.path.join(checkpoint_dir, f"int4_scales_{rank}.pt")
        if os.path.isfile(int4_scales_path):
            log_status(f"Rank {rank}: Loading int4 scales")
            int4_scales = torch.load(int4_scales_path, weights_only=True)

            def apply_quantization(key, weight):
                scale = int4_scales[key]
                return load_int4(
                    weight,
                    scale,
                    output_device=torch.device("cuda"),
                )

        else:
            log_status(f"Rank {rank}: Quantizing int4 weights from bf16")

            def apply_quantization(_, weight):
                return quantize_int4(weight, output_device=torch.device("cuda"))

    else:
        fp8_scales_path = os.path.join(checkpoint_dir, f"fp8_scales_{rank}.pt")
        if os.path.isfile(fp8_scales_path):
            log_status(f"Rank {rank}: Loading fp8 scales")
            fp8_scales = torch.load(fp8_scales_path, weights_only=True)

            def apply_quantization(key, weight):
                scale = fp8_scales[key]
                return load_fp8(
                    weight,
                    scale,
                    fp8_activation_scale_ub,
                    output_device=torch.device("cuda"),
                )

        else:
            log_status(f"Rank {rank}: Quantizing fp8 weights from bf16")

            def apply_quantization(_, weight):
                return quantize_fp8(weight, fp8_activation_scale_ub, output_device=torch.device("cuda"))

    processed_blocks = 0
    try:
        if use_rich_progress:
            progress.start()

        for _, block in model.named_modules():
            if not should_quantize_block(block):
                continue

            update_status(f"Rank {rank} - Layer {block.layer_id}")

            # Quantize only routed experts, not shared
            prefix = f"layers.{block.layer_id}.feed_forward"
            moe = block.feed_forward
            moe.experts.batched_swiglu = experts_batched_swiglu_wrapper.__get__(moe.experts)

            for key in ("w1", "w3", "w2"):
                param = getattr(moe.experts, key)
                update_status(f"Rank {rank} - Layer {block.layer_id} - MoE {key}")
                setattr(
                    moe.experts,
                    key,
                    apply_quantization(
                        f"{prefix}.experts.{key}",
                        param.transpose(1, 2).contiguous(),
                    ),
                )

            if quantization_mode == QuantizationMode.int4_mixed:
                # Quantize shared experts
                moe.shared_expert.forward = swiglu_wrapper_no_reduce.__get__(moe.shared_expert)
                for key in ("w1", "w3", "w2"):
                    param = getattr(moe.shared_expert, key)
                    update_status(f"Rank {rank} - Layer {block.layer_id} - MoE shared expert {key}")
                    param.weight = apply_quantization(f"{prefix}.shared_expert.{key}", param.weight)

            processed_blocks += 1
            update_status(message=None, completed=processed_blocks)

        update_status(f"Rank {rank} - Moving parameters to CUDA")

        param_count = 0
        for _, parameter in model.named_parameters():
            if not isinstance(parameter, Fp8ScaledWeights) and not isinstance(parameter, Int4ScaledWeights):
                parameter.data = parameter.to(device="cuda")
                param_count += 1

        update_status(f"Rank {rank} - Completed - moved {param_count} parameters to CUDA")
    finally:
        if use_rich_progress:
            progress.stop()

    return model


# fp8/int4 loading can be very slow so we add progress bars to make life slightly better
def logging_callbacks(
    use_rich_progress: bool,
    rank: int,
    model: Transformer,
    should_quantize_block: Callable[[nn.Module], bool],
):
    console = None
    if use_rich_progress:
        from rich.console import Console

        console = Console(highlight=False)

    def log_status(message: str) -> None:
        if use_rich_progress:
            console.print(message)
        elif rank == 0:  # Only log from rank 0 for non-rich logging
            log.info(message)

    total_blocks = sum(1 for _, block in model.named_modules() if should_quantize_block(block))
    progress = None
    if use_rich_progress:
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        progress = Progress(
            SpinnerColumn(),
            BarColumn(complete_style="green", finished_style="bright_green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(),
            TextColumn("[bold]{task.fields[status]}"),
            console=console,
            expand=True,
        )
        task_id = progress.add_task("[blue]Converting layers...", total=total_blocks, status="Starting")

    def update_status(message: str | None, completed: int | None = None) -> None:
        if use_rich_progress:
            if message is not None:
                progress.update(task_id, status=message)
            if completed is not None:
                progress.update(task_id, completed=completed)
        elif rank == 0 and completed and completed % 10 == 0:
            log.info(f"Rank {rank}: {completed}/{total_blocks} blocks completed")

    return progress, log_status, update_status
