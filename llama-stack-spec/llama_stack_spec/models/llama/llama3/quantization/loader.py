# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# type: ignore
import os
from typing import Any, cast

import torch
from fairscale.nn.model_parallel.initialize import get_model_parallel_rank
from fairscale.nn.model_parallel.layers import ColumnParallelLinear, RowParallelLinear
from fairscale.nn.model_parallel.mappings import reduce_from_model_parallel_region
from torch import Tensor, nn
from torchao.quantization.GPTQ import Int8DynActInt4WeightLinear

from ...datatypes import QuantizationMode
from ...quantize_impls import (
    Fp8ScaledWeights,
    ffn_swiglu,
    load_fp8,
    quantize_fp8,
)
from ..model import Transformer, TransformerBlock
from ..multimodal.model import CrossAttentionTransformer


def swiglu_wrapper(
    self,
    x: Tensor,
):
    out = ffn_swiglu(x, self.w1.weight, self.w3.weight, self.w2.weight)
    return reduce_from_model_parallel_region(out)


def convert_to_quantized_model(
    model: Transformer | CrossAttentionTransformer,
    checkpoint_dir: str,
    quantization_mode: str | None = None,
    fp8_activation_scale_ub: float | None = 1200.0,
    device: torch.device | None = None,
) -> Transformer | CrossAttentionTransformer:
    if quantization_mode == QuantizationMode.fp8_mixed:
        return convert_to_fp8_quantized_model(model, checkpoint_dir, fp8_activation_scale_ub, device)
    elif quantization_mode == QuantizationMode.int4_mixed:
        return convert_to_int4_quantized_model(model, checkpoint_dir, device)
    else:
        raise ValueError(f"Unsupported quantization mode: {quantization_mode}")


def convert_to_fp8_quantized_model(
    model: Transformer,
    checkpoint_dir: str,
    fp8_activation_scale_ub: float | None = 1200.0,
    device: torch.device | None = None,
) -> Transformer:
    # Move weights to GPU with quantization
    fp8_scales_path = os.path.join(checkpoint_dir, f"fp8_scales_{get_model_parallel_rank()}.pt")
    if os.path.isfile(fp8_scales_path):
        print("Loading fp8 scales...")
        fp8_scales = torch.load(fp8_scales_path, weights_only=True)

        for _, block in model.named_modules():
            if isinstance(block, TransformerBlock):
                if block.layer_id == 0 or block.layer_id == (model.n_layers - 1):
                    continue

                block.feed_forward.forward = swiglu_wrapper.__get__(block.feed_forward)
                for key in ("w1", "w3", "w2"):
                    param = getattr(block.feed_forward, key)
                    param.weight = load_fp8(
                        param.weight,
                        fp8_scales[f"{block.layer_id}_feed_forward.{key}_{get_model_parallel_rank()}"],
                        fp8_activation_scale_ub,
                    )
    else:
        print("Quantizing fp8 weights from bf16...")
        for _, block in model.named_modules():
            if isinstance(block, TransformerBlock):
                if block.layer_id == 0 or block.layer_id == (model.n_layers - 1):
                    continue
                block.feed_forward.forward = swiglu_wrapper.__get__(block.feed_forward)  # type: ignore
                for key in ("w1", "w3", "w2"):
                    param = getattr(block.feed_forward, key)
                    param.weight = quantize_fp8(
                        param.weight,
                        fp8_activation_scale_ub,
                        output_device=device,
                    )

    for _, parameter in model.named_parameters():
        if not isinstance(parameter, Fp8ScaledWeights):
            parameter.data = parameter.to(device=device)
    return model


class Int8DynActInt4WeightLinearLoRA(Int8DynActInt4WeightLinear):
    """
    Int8DynActInt4WeightLinear with LoRA adaptor.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to use bias.
        device: Device to use.
        group_size: Group size for quantization.
        precision: Precision of quantization.
        scales_precision: Precision of scales.
        lora_rank: Rank of LoRA adaptor.
        lora_scale: Scale of LoRA adaptor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=False,
        device=None,
        # quantization parameters
        group_size: int = 256,
        precision: torch.dtype = torch.float32,
        scales_precision: torch.dtype = torch.float32,
        # LoRA parameters
        lora_rank: int | None = None,
        lora_scale: float | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            groupsize=group_size,
            precision=precision,
            scales_precision=scales_precision,
        )
        self.lora_scale: float | None = None
        self.adaptor: nn.Sequential | None = None
        if lora_rank is not None:
            assert lora_scale is not None, "Please specify lora scale for LoRA."
            # Low-rank adaptation. See paper for more details: https://arxiv.org/abs/2106.09685
            self.adaptor = nn.Sequential()
            self.adaptor.add_module("A", nn.Linear(in_features, lora_rank, bias=False))
            self.adaptor.add_module("B", nn.Linear(lora_rank, out_features, bias=False))
            self.lora_scale = lora_scale
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """A hook to load the quantized weights from the state dict."""
        if prefix + "zeros" not in state_dict:
            # Zero-point may not be saved in the state dict. In this case, we assume it's zero.
            assert prefix + "scales" in state_dict
            state_dict[prefix + "zeros"] = torch.zeros_like(state_dict[prefix + "scales"])

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        module_out = super().forward(input_)
        if self.adaptor is not None:
            adaptor_out = self.adaptor(input_) * self.lora_scale
            return module_out + adaptor_out
        return module_out


class Int8WeightEmbedding(torch.nn.Embedding):
    """An embedding layer to load int8 weights.

    Args:
        num_embeddings: Number of embeddings.
        embedding_dim: Embedding dimension.
        padding_idx: Padding index.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        device=None,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx, device=device)

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """A hook to load the quantized embedding weight and scales from the state dict."""
        weights = state_dict.pop(prefix + "weight")
        scales = state_dict.pop(prefix + "scales")
        state_dict[prefix + "weight"] = weights * scales


class Int8WeightLinear(torch.nn.Linear):
    """A linear layer to load int8 weights.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to use bias.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None) -> None:
        super().__init__(in_features, out_features, bias, device=device)

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """A hook to load the quantized linear weight and scales from the state dict."""
        weights = state_dict.pop(prefix + "weight")
        scales = state_dict.pop(prefix + "scales")
        state_dict[prefix + "weight"] = weights * scales


def _prepare_model_int4_weight_int8_dynamic_activation(
    model: torch.nn.Module,
    group_size: int,
    lora_rank: int | None,
    lora_scale: float | None,
):
    """Prepare the model for int4 weight and int8 dynamic activation quantization.

    Note that the weights of embedding and output layers are quantized to int8.
    """
    device = None
    for module_name, module in model.named_children():
        if module_name == "output":
            quantized_module = Int8WeightLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                device=device,
            )
            del module
            setattr(model, module_name, quantized_module)
        elif module_name == "tok_embeddings":
            quantized_module = Int8WeightEmbedding(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                padding_idx=module.padding_idx,
                device=device,
            )
            del module
            setattr(model, module_name, quantized_module)
        elif isinstance(module, ColumnParallelLinear | RowParallelLinear | nn.Linear):
            quantized_module = Int8DynActInt4WeightLinearLoRA(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=False,
                group_size=group_size,
                lora_rank=lora_rank,
                lora_scale=lora_scale,
                device=device,
            )
            del module
            setattr(model, module_name, quantized_module)
        else:
            _prepare_model_int4_weight_int8_dynamic_activation(module, group_size, lora_rank, lora_scale)

    return model


def convert_to_int4_quantized_model(
    model: Transformer | CrossAttentionTransformer,
    checkpoint_dir: str,
    device: torch.device | None = None,
) -> Transformer | CrossAttentionTransformer:
    """Convert the model to int4 quantized model."""
    model_args = model.params
    assert model_args.quantization_args is not None, "Quantization args must be specified."
    quantization_args = model_args.quantization_args
    if quantization_args.scheme is None:
        raise ValueError("Quantization scheme must be specified in 'quantization_args'.")

    if quantization_args.scheme.value != "int4_weight_int8_dynamic_activation":
        raise NotImplementedError(
            "Only int4 quantization with 'int4_weight_int8_dynamic_activation' scheme is supported."
        )

    group_size = model_args.quantization_args.group_size
    if group_size is None:
        raise ValueError("'group_size' cannot be None in 'quantization_args'. Please specify it.")

    if model_args.lora_args is None:
        # Certain quantized models (e.g., SpinQuant) may not have LoRA.
        lora_rank = None
        lora_scale = None
    else:
        lora_rank = model_args.lora_args.rank
        lora_scale = model_args.lora_args.scale

    _prepare_model_int4_weight_int8_dynamic_activation(model, group_size, lora_rank, lora_scale)
    return cast(Transformer | CrossAttentionTransformer, model.to(device=device))
