# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import math
import re

import torch
from torch import nn


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Hadamard transform.

    This function performs the Hadamard transform on the input tensor 'x'.
    The Hadamard transform is a linear transformation that multiplies the input
    tensor by the Hadamard matrix of dimension n x n, where n is the size of
    the last dimension of the input tensor.
    """
    *_, n = x.shape
    m = int(math.log2(n))
    assert n == 1 << m, "n must be a power of 2"
    x = x[..., None]
    inv_sqrt2 = 0.5**0.5
    for _ in range(m):
        top = x[..., ::2, :] + x[..., 1::2, :]
        bot = x[..., ::2, :] - x[..., 1::2, :]
        x = torch.cat((top, bot), dim=-1)
        x *= inv_sqrt2
    res = x.squeeze(-2)
    return res


class HadamardModule(torch.nn.Module):
    """A module that applies the Hadamard transform to the input tensor.

    Args:
        group_size: The size of the groups that the input tensor will be divided into
            before applying the Hadamard transform.
    """

    def __init__(self, group_size: int) -> None:
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reshape_back = False
        orig_shape = x.shape
        if self.group_size != x.shape[-1]:
            reshape_back = True
            x = x.reshape(-1, x.shape[-1] // self.group_size, self.group_size)
        x = hadamard_transform(x)
        if reshape_back:
            x = x.reshape(orig_shape)
        return x


def add_hadamard_transform_for_spinquant(model: torch.nn.Module, prefix: str = "") -> None:
    """
    Adds a Hadamard transform to the last linear layer of each feedforward network (FFN) in the model.
    This function recursively traverses the model's children and looks for layers that match the pattern
    "layers.<digit>.feed_forward.w2", where <digit> is one or more digits. When such a layer is found,
    it is replaced with a new sequential module that consists of a HadamardModule followed by the original
    layer. The HadamardModule applies the Hadamard transform to the input tensor.

    See `SpinQuant <https://arxiv.org/abs/2405.16406>_` paper for more details.

    Args:
        model: An instance of 'torch.nn.Module' (e.g., Transformer model).
        prefix: A string prefix to add to the full name of each child module.

    Returns:
        None
    """

    pattern_last_linear_ffn = r"layers.\d+.feed_forward.w2"
    for module_name, module in model.named_children():
        child_full_name = prefix + "." + module_name
        if re.search(pattern_last_linear_ffn, child_full_name):
            # Module matching this pattern should be nn.Linear with in_features
            assert isinstance(module, nn.Linear), f"Expected nn.Linear, got {type(module)}"
            new_module = nn.Sequential(HadamardModule(group_size=module.in_features), module)
            del module
            setattr(model, module_name, new_module)
        else:
            add_hadamard_transform_for_spinquant(module, (prefix + "." if prefix else prefix) + module_name)
