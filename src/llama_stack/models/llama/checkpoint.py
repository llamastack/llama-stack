# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import concurrent.futures
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fairscale.nn.model_parallel.initialize import get_model_parallel_rank, get_model_parallel_world_size


def map_mp_rank(old_mp_size: int, new_mp_size: int, new_mp_rank: int) -> list[int]:
    """Map a new MP rank to a list of old MP ranks given a change in MP size."""
    if new_mp_size % old_mp_size == 0:
        # Read old MP shard and split it into smaller ones
        return [new_mp_rank * old_mp_size // new_mp_size]
    elif old_mp_size % new_mp_size == 0:
        # Merge old MP shards into a single one
        mp_factor = old_mp_size // new_mp_size
        return list(range(new_mp_rank * mp_factor, (new_mp_rank + 1) * mp_factor))
    else:
        raise ValueError(
            f"Either old MP size or new MP size should be a multiple of the other: "
            f"{old_mp_size} % {new_mp_size} != 0 and {new_mp_size} % {old_mp_size} != 0"
        )


def maybe_reshard_state_dict(
    ckpt_paths: list[Path],
    n_kv_heads: int,
    moe_num_experts: int | None = None,
    map_location: str | torch.device = "cpu",
    mmap: bool = True,
) -> dict[str, torch.Tensor]:
    if str(map_location) == "cpu":
        torch.set_default_dtype(torch.bfloat16)
    else:
        torch.set_default_dtype(torch.bfloat16)

    ckpt_paths_array = np.array(sorted(ckpt_paths))

    new_mp_size, new_mp_rank = get_model_parallel_world_size(), get_model_parallel_rank()
    old_mp_size = len(ckpt_paths_array)
    old_mp_ranks = map_mp_rank(old_mp_size, new_mp_size, new_mp_rank)

    print(f"Loading checkpoint shards:\n{str(ckpt_paths_array[old_mp_ranks])}")  # type: ignore
    paths = ckpt_paths_array[old_mp_ranks]  # type: ignore
    state_dicts = [torch.load(str(p), map_location=map_location, mmap=mmap) for p in paths]

    if new_mp_size == old_mp_size:
        return state_dicts[0]  # type: ignore

    if moe_num_experts is not None:
        state_dicts = [convert_moe_weights(d, moe_num_experts) for d in state_dicts]

    print(f"Resharding {len(state_dicts)} state dicts from MP size {old_mp_size} to MP size {new_mp_size}")
    return reshard_mp(
        state_dicts,
        size=max(new_mp_size // old_mp_size, 1),
        rank=new_mp_rank % max(new_mp_size // old_mp_size, 1),
        repeat_qk_qv=max(new_mp_size // n_kv_heads, 1),
    )


_WEIGHT_ROW_KEY = {
    "feed_forward.w2",
    "feed_forward.mlp.fc2",
    "attention.wo",
    "feed_forward.mlp.fc2_weight",
    "feed_forward.w_out_shared_DF.weight",
    "attn.wo.weight",
    "mlp.c_proj.weight",
}
_MOE_WEIGHT_ROW_KEY = {"feed_forward.experts.(moe_w_in_eD_F|moe_w_swiglu_eD_F)"}

_WEIGHT_COLUMN_KEY = {
    "output",
    "feed_forward.(w1|w3)",
    "feed_forward.mlp.(fc1|fc3)",
    "feed_forward.mlp.fc1_weight",
    "attention.(wk|wq|wv|wqkv).weight",
    "feed_forward.(w_in_shared_FD|w_swiglu_FD)",
    "attn.(wk|wq|wv).weight",
    "attn.(wk|wq|wv).bias",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "conv1._linear.weight",
    "tok_embeddings.weight",
    "vision_projection.weight",
}
_MOE_WEIGHT_COLUMN_KEY = {"feed_forward.experts.moe_w_out_eF_D"}


def reshard_mp(
    state_dicts: list[dict[str, torch.Tensor]],
    size: int,
    rank: int,
    repeat_qk_qv: int = 1,
) -> dict[str, torch.Tensor]:
    """
    Reshard a list of state dicts into a single state dict given a change in MP size.
    If the list has more than one state dict, we concatenate the values of the same
    key across all state dicts. Otherwise, we just slice it for the current MP rank.
    """

    def concat_or_chunk(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
        if len(tensors) > 1:
            return torch.cat(tensors, dim=dim)
        return tensors[0].chunk(size, dim=dim)[rank].clone()

    def process_key(key: str) -> torch.Tensor:
        if row_regex.search(key):
            return concat_or_chunk([s[key] for s in state_dicts], dim=-1)
        elif column_regex.search(key):
            if "w13" in key or "fc1_weight" in key:
                dims = state_dicts[0][key].size()
                values = [s[key].view(2, dims[0] // 2, *dims[1:]) for s in state_dicts]
                return concat_or_chunk(values, dim=1).flatten(0, 1)
            elif "qkv" in key:
                q_dim = state_dicts[0][key.replace("qkv", "o")].size(1)
                kv_dim = (state_dicts[0][key].size(0) - q_dim) // 2
                values = [s[key].split((q_dim, kv_dim, kv_dim)) for s in state_dicts]
                return torch.cat([concat_or_chunk(x, dim=0) for x in zip(*values, strict=False)])  # type: ignore
            elif "wk.weight" in key or "wv.weight" in key:
                # Support MP > #kv_head
                return concat_or_chunk([s[key].repeat(repeat_qk_qv, 1) for s in state_dicts], dim=0)
            elif key == "output.bias" or key == "fc.weight":
                return concat_or_chunk([s[key] for s in state_dicts], dim=0)
            elif "w_" in key:
                return concat_or_chunk([s[key] for s in state_dicts], dim=-2)
            else:
                return concat_or_chunk([s[key] for s in state_dicts], dim=0)
        else:
            return state_dicts[0][key].clone()

    row_keys = _WEIGHT_ROW_KEY | _MOE_WEIGHT_ROW_KEY
    column_keys = _WEIGHT_COLUMN_KEY | _MOE_WEIGHT_COLUMN_KEY

    column_regex = re.compile("|".join(column_keys))
    row_regex = re.compile("|".join(row_keys))

    output: dict[str, torch.Tensor] = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Note: only processes keys in the first state dict.
        # Assumes keys are the same across all state dicts.
        mappings = {executor.submit(process_key, key): key for key in state_dicts[0]}
        for future in concurrent.futures.as_completed(mappings):
            output[mappings[future]] = future.result()
    return output


def convert_moe_weights(state_dict: dict[str, Any], num_experts: int) -> dict[str, Any]:
    routed_keys = _MOE_WEIGHT_ROW_KEY | _MOE_WEIGHT_COLUMN_KEY
    routed_regex = re.compile("|".join(routed_keys))
    keys = list(state_dict.keys())
    for key in keys:
        if routed_regex.search(key):
            state_dict[key] = state_dict.pop(key).unflatten(0, (num_experts, -1)).squeeze(dim=0)
    return state_dict
