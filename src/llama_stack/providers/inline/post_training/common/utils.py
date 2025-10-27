# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import gc


def evacuate_model_from_device(model, device: str):
    """Safely clear a model from memory and free device resources.
    This function handles the proper cleanup of a model by:
    1. Moving the model to CPU if it's on a non-CPU device
    2. Deleting the model object to free memory
    3. Running garbage collection
    4. Clearing CUDA cache if the model was on a CUDA device
    Args:
        model: The PyTorch model to clear
        device: The device type the model is currently on ('cuda', 'mps', 'cpu')
    Note:
        - For CUDA devices, this will clear the CUDA cache after moving the model to CPU
        - For MPS devices, only moves the model to CPU (no cache clearing available)
        - For CPU devices, only deletes the model object and runs garbage collection
    """
    if device != "cpu":
        model.to("cpu")

    del model
    gc.collect()

    if device == "cuda":
        # we need to import such that this is only imported when the method is called
        import torch

        torch.cuda.empty_cache()
