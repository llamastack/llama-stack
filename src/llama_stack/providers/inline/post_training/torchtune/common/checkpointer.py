# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torchtune import training
from torchtune.models import convert_weights
from torchtune.training.checkpointing._utils import (
    ADAPTER_CONFIG_FNAME,
    ADAPTER_MODEL_FNAME,
    REPO_ID_FNAME,
    SUFFIXES_TO_NOT_COPY,
    ModelType,
    copy_files,
    safe_torch_load,
)
from torchtune.utils._logging import get_logger

logger = get_logger("DEBUG")


class TorchtuneCheckpointer:
    def __init__(
        self,
        model_id: str,
        training_algorithm: str,
        checkpoint_dir: str,
        checkpoint_files: list[str],
        output_dir: str,
        model_type: str,
    ):
        # Fail fast if ``checkpoint_files`` is invalid
        # TODO: support loading more than one file
        if len(checkpoint_files) != 1:
            raise ValueError(
                "Currently we only support reading from a single torchtune checkpoint file. "
                f"Got {len(checkpoint_files)} files instead."
            )
        self._checkpoint_file = checkpoint_files[0]
        self._model_id = model_id
        self._training_algorithm = training_algorithm
        self._checkpoint_dir = Path(checkpoint_dir)
        self._model_type = ModelType[model_type]
        self._output_dir = output_dir
        # get ckpt paths
        self._checkpoint_path = Path.joinpath(self._checkpoint_dir, self._checkpoint_file)

    def load_checkpoint(self) -> dict[str, Any]:
        """
        Load Meta checkpoint from file. Currently only loading from a single file is supported.
        """
        state_dict: dict[str, Any] = {}
        model_state_dict = safe_torch_load(self._checkpoint_path)
        if self._model_type == ModelType.LLAMA3_VISION:
            from torchtune.models.llama3_2_vision._convert_weights import (
                llama3_vision_meta_to_tune,
            )

            state_dict[training.MODEL_KEY] = llama3_vision_meta_to_tune(model_state_dict)
        else:
            state_dict[training.MODEL_KEY] = convert_weights.meta_to_tune(model_state_dict)

        # llama3_2 has tied weights, so we need to remove the output.weight key
        if self._model_type == ModelType.LLAMA3_2:
            logger.info(
                "Identified model_type = Llama3_2. Ignoring output.weight in"
                " checkpoint in favor of the tok_embedding.weight"
                " tied weights."
            )
            state_dict[training.MODEL_KEY].pop("output.weight")

        return state_dict

    def save_checkpoint(
        self,
        state_dict: dict[str, Any],
        epoch: int,
        adapter_only: bool = False,
        checkpoint_format: str | None = None,
    ) -> str:
        model_file_path = Path(self._output_dir) / f"{self._model_id}-{self._training_algorithm}-{epoch}"
        if checkpoint_format == "meta" or checkpoint_format is None:
            self._save_meta_format_checkpoint(model_file_path, state_dict, adapter_only)
        elif checkpoint_format == "huggingface":
            # Note: for saving hugging face format checkpoints, we only support saving adapter weights now
            self._save_hf_format_checkpoint(model_file_path, state_dict)
        else:
            raise ValueError(f"Unsupported checkpoint format: {format}")
        return str(model_file_path)

    def _save_meta_format_checkpoint(
        self,
        model_file_path: Path,
        state_dict: dict[str, Any],
        adapter_only: bool = False,
    ) -> None:
        model_file_path.mkdir(parents=True, exist_ok=True)

        # copy the related files for inference
        source_path = Path.joinpath(self._checkpoint_dir, "params.json")
        if source_path.exists():
            shutil.copy(
                source_path,
                Path.joinpath(model_file_path, "params.json"),
            )
        source_path = Path.joinpath(self._checkpoint_dir, "tokenizer.model")
        if source_path.exists():
            shutil.copy(
                source_path,
                Path.joinpath(model_file_path, "tokenizer.model"),
            )
        source_path = Path.joinpath(self._checkpoint_dir, "orig_params.json")
        if source_path.exists():
            shutil.copy(
                source_path,
                Path.joinpath(model_file_path, "orig_params.json"),
            )

        if not adapter_only:
            model_state_dict = state_dict[training.MODEL_KEY]
            if self._model_type == ModelType.LLAMA3_VISION:
                from torchtune.models.llama3_2_vision._convert_weights import (
                    llama3_vision_tune_to_meta,
                )

                state_dict[training.MODEL_KEY] = llama3_vision_tune_to_meta(model_state_dict)
            else:
                # llama3_2 has tied weights, so we need to add the output.weight key
                if self._model_type == ModelType.LLAMA3_2 and "output.weight" not in model_state_dict:
                    model_state_dict["output.weight"] = model_state_dict["tok_embeddings.weight"]

                state_dict[training.MODEL_KEY] = convert_weights.tune_to_meta(model_state_dict)

            model_file_name = Path.joinpath(model_file_path, "consolidated.00.pth")

            torch.save(state_dict[training.MODEL_KEY], model_file_name)
            logger.info(
                "Model checkpoint of size "
                f"{os.path.getsize(model_file_name) / 1000**3:.2f} GB "
                f"saved to {model_file_name}"
            )

        if training.ADAPTER_KEY in state_dict:
            adapter_file_path = model_file_path / "adapter"
            adapter_file_path.mkdir(parents=True, exist_ok=True)
            adapter_file_name = Path.joinpath(adapter_file_path, "adapter.pth")
            torch.save(state_dict[training.ADAPTER_KEY], adapter_file_name)
            logger.info(
                "Adapter checkpoint of size "
                f"{os.path.getsize(adapter_file_name) / 1000**3:.2f} GB "
                f"saved to {adapter_file_name}"
            )

        elif adapter_only:
            raise ValueError(
                "Adapter checkpoint not found in state_dict. Please ensure that the state_dict contains adapter weights."
            )

    def _save_hf_format_checkpoint(
        self,
        model_file_path: Path,
        state_dict: dict[str, Any],
    ) -> None:
        # the config.json file contains model params needed for state dict conversion
        config = json.loads(Path.joinpath(self._checkpoint_dir.parent, "config.json").read_text())

        # repo_id is necessary for when saving an adapter config, so its compatible with HF.
        # This json file is produced and saved in the download step.
        # contents are {"repo_id": "some_model/some_model_version"}
        repo_id_path = Path.joinpath(self._checkpoint_dir.parent, REPO_ID_FNAME).with_suffix(".json")
        self.repo_id = None
        if repo_id_path.exists():
            with open(repo_id_path) as json_file:
                data = json.load(json_file)
                self.repo_id = data.get("repo_id")

        if training.ADAPTER_KEY in state_dict:
            # TODO: saving it "as is" is a requirement because, if we only save with
            # convert_weights.tune_to_peft_adapter_weights, we do NOT have a fn
            # convert_weights.peft_to_tune. The .pt format is not needed, but
            # it is an easy way to distinguish the adapters. Ideally we should save only one.
            output_path = Path.joinpath(model_file_path, ADAPTER_MODEL_FNAME).with_suffix(".pt")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict[training.ADAPTER_KEY], output_path)
            logger.info(
                f"Adapter checkpoint of size {os.path.getsize(output_path) / 1024**3:.2f} GiB saved to {output_path}"
            )

            state_dict[training.ADAPTER_KEY] = convert_weights.tune_to_peft_adapter_weights(
                state_dict[training.ADAPTER_KEY],
                num_heads=config["num_attention_heads"],
                num_kv_heads=config["num_key_value_heads"],
                dim=config["hidden_size"],
                head_dim=config.get("head_dim", None),
            )
            output_path = Path.joinpath(model_file_path, "adapter", ADAPTER_MODEL_FNAME)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path = output_path.with_suffix(".safetensors")
            save_file(
                state_dict[training.ADAPTER_KEY],
                output_path,
                metadata={"format": "pt"},
            )
            logger.info(
                f"Adapter checkpoint of size {os.path.getsize(output_path) / 1024**3:.2f} GiB saved to {output_path}"
            )
        else:
            raise ValueError(
                "Adapter checkpoint not found in state_dict. Please ensure that the state_dict contains adapter weights."
            )

        if training.ADAPTER_CONFIG in state_dict:
            state_dict[training.ADAPTER_CONFIG] = convert_weights.tune_to_peft_adapter_config(
                adapter_config=state_dict[training.ADAPTER_CONFIG],
                base_model_name_or_path=self.repo_id,
            )

            output_path = Path.joinpath(model_file_path, "adapter", ADAPTER_CONFIG_FNAME).with_suffix(".json")
            with open(output_path, "w") as f:
                json.dump(state_dict[training.ADAPTER_CONFIG], f)
            logger.info(
                f"Adapter checkpoint of size {os.path.getsize(output_path) / 1024**3:.2f} GiB saved to {output_path}"
            )

        # Save all files in ckpt_dir, except model weights and mapping, to output_dir/epoch_{epoch}
        # So its easy to run inference with the model using this epoch's checkpoint
        copy_files(
            self._checkpoint_dir.parent,
            model_file_path,
            ignore_suffixes=SUFFIXES_TO_NOT_COPY,
        )
