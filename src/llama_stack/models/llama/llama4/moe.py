# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# ruff: noqa: N806
# pyre-strict
from typing import Any

import fairscale.nn.model_parallel.initialize as fs_init
import torch
from fairscale.nn.model_parallel.mappings import reduce_from_model_parallel_region
from torch import Tensor, nn
from torch.nn import functional as F

from .args import MoEArgs
from .ffn import FeedForward


class Experts(nn.Module):
    def __init__(
        self,
        num_local_experts: int,
        dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        dtype = torch.get_default_dtype()
        self.num_local_experts = num_local_experts
        self.dim = dim
        divide_factor = fs_init.get_model_parallel_world_size()

        self.w1: nn.Parameter = nn.Parameter(
            torch.empty(
                num_local_experts,
                dim,
                divide_exact(hidden_dim, divide_factor),
                dtype=dtype,
            )
        )

        self.w2: nn.Parameter = nn.Parameter(
            torch.empty(
                num_local_experts,
                divide_exact(hidden_dim, divide_factor),
                dim,
                dtype=dtype,
            )
        )

        self.w3: nn.Parameter = nn.Parameter(
            torch.empty(
                num_local_experts,
                dim,
                divide_exact(hidden_dim, divide_factor),
                dtype=dtype,
            )
        )

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
        self.prefix = prefix
        if prefix + "moe_w_in_eD_F" in state_dict:
            e = self.num_local_experts
            D = self.dim
            state_dict[prefix + "w1"] = state_dict.pop(prefix + "moe_w_in_eD_F").view(e, D, -1)
            state_dict[prefix + "w2"] = state_dict.pop(prefix + "moe_w_out_eF_D").view(e, -1, D)
            state_dict[prefix + "w3"] = state_dict.pop(prefix + "moe_w_swiglu_eD_F").view(e, D, -1)

    def forward(
        self,
        routed_in_egD: torch.Tensor,  # noqa: N803
    ) -> torch.Tensor:
        e = self.num_local_experts
        D = self.dim

        x_egD = routed_in_egD.view(e, -1, D)

        out_egD = self.batched_swiglu(x_egD, self.w1, self.w3, self.w2)
        out_egD = out_egD.view(-1, D)

        return out_egD

    def batched_swiglu(self, x: Tensor, w1: Tensor, w3: Tensor, w2: Tensor) -> Tensor:
        middle_out_egF = F.silu(torch.bmm(x, w1)) * torch.bmm(x, w3)
        return torch.bmm(middle_out_egF, w2)


class MoE(torch.nn.Module):
    """
    Tensors used in this module are annotated with the suffixes that indicate the shape of the tensor.
    Several commonly used annotations include:
    - a: bsz*slen
    - E: number of experts
    - e: number of local experts per ep (n_experts/ep)
    - D: hidden dimension
    - d: D/tp
    - F: model dimension
    - G: number of tokens per expert (a * capacity_factor / E)
    - g: number of tokens per expert per TP rank (i.e., G/TP)

    Examples:
    x_aD [a, D]
    routed_in_etG_D [et*G, D]
    x_eGD: [e, G, D]
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        ffn_dim_multiplier: float,
        multiple_of: int,
        moe_args: MoEArgs,
    ) -> None:
        super().__init__()

        self.moe_args = moe_args

        hidden_dim_denom: float = 1
        if moe_args.auto_scale_F:
            hidden_dim_denom = moe_args.capacity_factor + 1

        hidden_dim = int(2 * hidden_dim / 3)

        # custom dim factor multiplier
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        if moe_args.auto_scale_F:
            hidden_dim = int(hidden_dim / hidden_dim_denom)

        hidden_dim += -hidden_dim % multiple_of

        num_local_experts: int = moe_args.num_experts
        dtype: torch.dtype = torch.get_default_dtype()
        self.experts = Experts(
            num_local_experts,
            dim,
            hidden_dim,
        )

        self.router_DE: nn.Parameter = nn.Parameter(torch.empty(dim, moe_args.num_experts, dtype=dtype))
        self.shared_expert = FeedForward(dim, hidden_dim, do_reduce=False)

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
        if prefix + "w_in_shared_FD.weight" in state_dict:
            state_dict[prefix + "shared_expert.w1.weight"] = state_dict.pop(prefix + "w_in_shared_FD.weight")
            state_dict[prefix + "shared_expert.w3.weight"] = state_dict.pop(prefix + "w_swiglu_FD.weight")
            state_dict[prefix + "shared_expert.w2.weight"] = state_dict.pop(prefix + "w_out_shared_DF.weight")

    def forward(self, x_bsD: Tensor) -> Tensor:  # noqa: N803
        _, slen, D = x_bsD.shape
        x_aD = x_bsD.view(-1, D)

        a = x_aD.shape[0]

        router_scores: Tensor = torch.matmul(x_aD, self.router_DE).transpose(0, 1)

        router_scores_aK, router_indices_aK = torch.topk(router_scores.transpose(0, 1), self.moe_args.top_k, dim=1)
        router_scores = (
            torch.full_like(router_scores.transpose(0, 1), float("-inf"))
            .scatter_(1, router_indices_aK, router_scores_aK)
            .transpose(0, 1)
        )
        router_indices = torch.arange(a, device=x_aD.device).view(1, -1).expand(router_scores.size(0), -1)

        router_scores = torch.sigmoid(router_scores)

        routed_in_EG_D: Tensor = torch.gather(
            x_aD,
            dim=0,
            index=router_indices.reshape(-1, 1).expand(-1, D),
        )
        routed_in_EG_D = routed_in_EG_D * router_scores.reshape(-1, 1)

        out_aD = self.shared_expert(x_aD)
        routed_out_eg_D = self.experts(routed_in_EG_D.detach())

        router_indices_EG_D = router_indices.reshape(-1, 1).expand(-1, D)
        out_aD.scatter_add_(
            dim=0,
            index=router_indices_EG_D,
            src=routed_out_eg_D.view(-1, D),
        )
        out_aD = reduce_from_model_parallel_region(out_aD)
        return out_aD.view(-1, slen, D)


def divide_exact(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"
    return numerator // denominator
