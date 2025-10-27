# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import math
from typing import Any

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn

from .args import ModelArgs
from .datatypes import TransformerInput, TransformerOutput
from .ffn import FeedForward
from .moe import MoE


def rmsnorm(x, eps):
    def _norm(y):
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)

    return _norm(x.float()).type_as(x)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return rmsnorm(x, self.eps) * self.weight


def apply_scaling(freqs: torch.Tensor, scale_factor: float, high_freq_factor: float):
    low_freq_factor = 1
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float,
    use_scaled: bool,
    scale_factor: float,
    high_freq_factor: float,
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs, scale_factor, high_freq_factor)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    # TODO: this module needs to be moved into a separate file since it can be used by
    # the vision encoder as well.
    def __init__(
        self,
        args: ModelArgs,
        use_qk_norm: bool,
        use_rope: bool,
        add_bias: bool = False,
    ):
        super().__init__()
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        # For attention temperature tuning
        self.attn_temperature_tuning = args.attn_temperature_tuning
        self.floor_scale = args.floor_scale
        self.attn_scale = args.attn_scale

        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        world_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // world_size
        self.n_local_kv_heads = self.n_kv_heads // world_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=add_bias,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=add_bias,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=add_bias,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=add_bias,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.norm_eps = args.norm_eps
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
        if prefix + "wqkv.weight" in state_dict:
            wqkv = state_dict.pop(prefix + "wqkv.weight")
            d, r = divmod(wqkv.shape[0], self.n_heads + 2 * self.n_kv_heads)
            if r != 0:
                raise ValueError(
                    f"shape={tuple(wqkv.shape)} is not divisible by "
                    f"n_heads ({self.n_heads}) + 2 * n_kv_heads ({self.n_kv_heads})"
                )
            wq, wk, wv = wqkv.split([d * self.n_heads, d * self.n_kv_heads, d * self.n_kv_heads], dim=0)
            state_dict[prefix + "wq.weight"] = wq
            state_dict[prefix + "wk.weight"] = wk
            state_dict[prefix + "wv.weight"] = wv

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if self.use_rope:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.use_qk_norm:
            xq = rmsnorm(xq, self.norm_eps)
            xk = rmsnorm(xk, self.norm_eps)

        # We are applying temperature tuning (https://arxiv.org/abs/2501.19399) to NoPE layers, where
        # the inference-time temperature tuning function is customized to not affect short context
        # while working at very long context
        if self.attn_temperature_tuning and not self.use_rope:
            seq_positions = torch.arange(start_pos, start_pos + seqlen, device=xq.device, dtype=torch.float32)
            attn_scales = torch.log(torch.floor((seq_positions + 1.0) / self.floor_scale) + 1.0) * self.attn_scale + 1.0

            # reshape for broadcasting [seqlen] -> [1, seqlen, 1, 1]
            attn_scales = attn_scales.view(1, seqlen, 1, 1)
            xq = xq * attn_scales

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        xk = self.cache_k[:bsz, : start_pos + seqlen]
        xv = self.cache_v[:bsz, : start_pos + seqlen]

        xq, xk, xv = [t.transpose(1, 2) for t in (xq, xk, xv)]

        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)

        attn_output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, dropout_p=0.0)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(attn_output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads if args.head_dim is None else args.head_dim

        self.is_nope_layer = args.nope_layer_interval is not None and (layer_id + 1) % args.nope_layer_interval == 0

        use_rope = not self.is_nope_layer
        use_qk_norm = args.use_qk_norm and not self.is_nope_layer

        self.attention = Attention(args, use_rope=use_rope, use_qk_norm=use_qk_norm)

        if args.moe_args and (layer_id + 1) % args.moe_args.interleave_moe_layer_step == 0:
            self.feed_forward = MoE(
                dim=args.dim,
                hidden_dim=int(args.ffn_exp * args.dim),
                ffn_dim_multiplier=args.ffn_dim_multiplier,
                multiple_of=args.multiple_of,
                moe_args=args.moe_args,
            )
        else:
            hidden_dim = int(4 * args.dim)
            hidden_dim = int(2 * hidden_dim / 3)
            if args.ffn_dim_multiplier is not None:
                hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
            hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=hidden_dim,
            )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

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
        if prefix + "attention.wqkv.layer_norm_weight" in state_dict:
            state_dict[prefix + "attention_norm.weight"] = state_dict.pop(prefix + "attention.wqkv.layer_norm_weight")

        if prefix + "feed_forward.mlp.layer_norm_weight" in state_dict:
            state_dict[prefix + "ffn_norm.weight"] = state_dict.pop(prefix + "feed_forward.mlp.layer_norm_weight")
        elif prefix + "feed_forward.norm.weight" in state_dict:
            state_dict[prefix + "ffn_norm.weight"] = state_dict.pop(prefix + "feed_forward.norm.weight")

        for k in (
            "feed_forward.experts.mlp",
            "feed_forward.mlp_shared",
            "attention.wo",
            "attention.wqkv",
        ):
            if prefix + k + "._extra_state" in state_dict:
                state_dict.pop(prefix + k + "._extra_state")

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        global_attn_mask: torch.Tensor | None,
        local_attn_mask: torch.Tensor | None,
    ):
        # The iRoPE architecture uses global attention mask for NoPE layers or
        # if chunked local attention is not used
        if self.is_nope_layer or local_attn_mask is None:
            mask = global_attn_mask
        else:
            mask = local_attn_mask

        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, **kwargs) -> None:
        super().__init__()
        self.args = args

        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = VocabParallelEmbedding(args.vocab_size, args.dim, init_method=lambda x: x)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = ColumnParallelLinear(args.dim, args.vocab_size, bias=False, init_method=lambda x: x)

        self.freqs_cis = precompute_freqs_cis(
            args.dim // args.n_heads,
            args.max_seq_len * 2,
            args.rope_theta,
            args.use_scaled_rope,
            args.rope_scaling_factor,
            args.rope_high_freq_factor,
        )
        vision_args = self.args.vision_args
        if vision_args:
            # circular import otherwise until we refactor out Attention
            from .vision.embedding import VisionEmbeddings

            self.vision_embeddings = VisionEmbeddings(vision_args)
            self.vision_projection = ColumnParallelLinear(
                vision_args.output_dim,
                args.dim,
                bias=False,
                init_method=lambda x: x,
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
        if prefix + "rope.freqs" in state_dict:
            state_dict.pop(prefix + "rope.freqs")

    @torch.inference_mode()
    def forward(self, model_input: TransformerInput) -> TransformerOutput:
        tokens = model_input.tokens
        start_pos = model_input.tokens_position
        assert isinstance(start_pos, int), (
            "This implementation does not support different start positions per batch item"
        )

        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        if image_embedding := model_input.image_embedding:
            h_image = self.vision_projection(image_embedding.embedding)
            h = h * ~image_embedding.mask + h_image * image_embedding.mask

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        global_attn_mask, local_attn_mask = None, None
        if seqlen > 1:
            global_attn_mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            global_attn_mask = torch.triu(global_attn_mask, diagonal=1).type_as(h)

            # https://github.com/pytorch/pytorch/issues/100005
            # torch.triu is buggy when the device is mps: filled values are
            # nan instead of 0.
            if global_attn_mask.device.type == torch.device("mps").type:
                global_attn_mask = torch.nan_to_num(global_attn_mask, nan=0.0)

            if chunk_size := self.args.attention_chunk_size:
                local_attn_mask = create_chunked_attention_mask(seqlen, chunk_size, tokens.device)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, global_attn_mask, local_attn_mask)
        h = self.norm(h)
        output = self.output(h).float()

        return TransformerOutput(logits=output)


# tokens (0, K), (K, 2K), (2K, 3K) attend to each other when doing local chunked attention
# in the iRoPE architecture
def create_chunked_attention_mask(seq_len: int, attention_chunk_size: int, device: torch.device) -> torch.Tensor:
    block_pos = torch.abs(
        (torch.arange(seq_len).unsqueeze(0) // attention_chunk_size)
        - (torch.arange(seq_len).unsqueeze(1) // attention_chunk_size)
    )
    token_pos = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
    mask = (block_pos == 0) & (token_pos <= 0)
    return mask.to(device)
