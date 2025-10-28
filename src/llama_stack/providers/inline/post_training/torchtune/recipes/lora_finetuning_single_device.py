# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import modules, training
from torchtune import utils as torchtune_utils
from torchtune.data import padded_collate_sft
from torchtune.models.llama3._tokenizer import Llama3Tokenizer
from torchtune.modules.loss import CEWithChunkedOutputLoss
from torchtune.modules.peft import (
    get_adapter_params,
    get_adapter_state_dict,
    get_lora_module_names,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)
from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup
from torchtune.training.metric_logging import DiskLogger
from tqdm import tqdm

from llama_stack.apis.common.training_types import PostTrainingMetric
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.post_training import (
    Checkpoint,
    DataConfig,
    LoraFinetuningConfig,
    OptimizerConfig,
    QATFinetuningConfig,
    TrainingConfig,
)
from llama_stack.core.utils.config_dirs import DEFAULT_CHECKPOINT_DIR
from llama_stack.core.utils.model_utils import model_local_dir
from llama_stack.log import get_logger
from llama_stack.models.llama.sku_list import resolve_model
from llama_stack.providers.inline.post_training.common.utils import evacuate_model_from_device
from llama_stack.providers.inline.post_training.torchtune.common import utils
from llama_stack.providers.inline.post_training.torchtune.common.checkpointer import (
    TorchtuneCheckpointer,
)
from llama_stack.providers.inline.post_training.torchtune.config import (
    TorchtunePostTrainingConfig,
)
from llama_stack.providers.inline.post_training.torchtune.datasets.sft import SFTDataset

log = get_logger(name=__name__, category="post_training")


class LoraFinetuningSingleDevice:
    # This recipe doesn't include several training efficiency setting within origin torchtune repo, including
    # - compile
    # - activation offloading

    # Resume from checkpoint hasn't been supported yet
    # Validation hasn't been supported yet

    # Currently logging only logs limited training metrics to local disk
    # will figure out more loggings and how it works with telemetry in future PRs

    _checkpointer: TorchtuneCheckpointer

    def __init__(
        self,
        config: TorchtunePostTrainingConfig,
        job_uuid: str,
        training_config: TrainingConfig,
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
        model: str,
        checkpoint_dir: str | None,
        algorithm_config: LoraFinetuningConfig | QATFinetuningConfig | None,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
    ) -> None:
        assert isinstance(training_config.data_config, DataConfig), "DataConfig must be initialized"

        self.job_uuid = job_uuid
        self.training_config = training_config
        if not isinstance(algorithm_config, LoraFinetuningConfig):
            raise ValueError("You need to speicifc LoraFinetuningConfig for LoRA finetuning")
        self.algorithm_config = algorithm_config
        self._device = torchtune_utils.get_device()
        self._dtype = training.get_dtype(training_config.dtype, device=self._device)
        self.model_id = model

        def model_checkpoint_dir(model) -> str:
            checkpoint_dir = Path(model_local_dir(model.descriptor()))

            paths = [Path(checkpoint_dir / f"consolidated.{ext}") for ext in ["pth", "00.pth"]]
            if not any(p.exists() for p in paths):
                checkpoint_dir = checkpoint_dir / "original"

            hf_repo = model.huggingface_repo or f"meta-llama/{model.descriptor()}"
            assert checkpoint_dir.exists(), (
                f"Could not find checkpoints in: {model_local_dir(model.descriptor())}. "
                f"Please download the model using `huggingface-cli download {hf_repo} --local-dir ~/.llama/{model.descriptor()}`"
            )
            return str(checkpoint_dir)

        if checkpoint_dir and checkpoint_dir != "null":
            self.checkpoint_dir = checkpoint_dir
        else:
            model_obj = resolve_model(self.model_id)
            if model_obj is None:
                raise ValueError(f"{self.model_id} not found. Your model id should be in the llama models SKU list")
            self.checkpoint_dir = model_checkpoint_dir(model_obj)

        self._output_dir = str(DEFAULT_CHECKPOINT_DIR)
        self._checkpoint_format = config.checkpoint_format

        self.seed = training.set_seed(seed=config.torch_seed)
        self.epochs_run = 0
        self.total_epochs = training_config.n_epochs
        self._data_format = training_config.data_config.data_format
        self._shuffle = training_config.data_config.shuffle
        self._batch_size = training_config.data_config.batch_size
        self._train_on_input = training_config.data_config.train_on_input

        # this is important for debugging purpose
        self.max_steps_per_epoch = training_config.max_steps_per_epoch
        self.global_step = 0

        self._gradient_accumulation_steps = training_config.gradient_accumulation_steps
        self.max_validation_steps = training_config.max_validation_steps

        self._clip_grad_norm = 1.0

        self._enable_activation_checkpointing = False
        self._enable_activation_offloading = False
        if training_config.efficiency_config:
            if training_config.efficiency_config.enable_activation_checkpointing:
                self._enable_activation_checkpointing = (
                    training_config.efficiency_config.enable_activation_checkpointing
                )
            if training_config.efficiency_config.enable_activation_offloading:
                self._enable_activation_offloading = training_config.efficiency_config.enable_activation_offloading

        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api

    async def load_checkpoint(self):
        def get_checkpoint_files(checkpoint_dir: str) -> list[str]:
            try:
                # List all files in the given directory
                files = os.listdir(checkpoint_dir)
                # Filter files that end with .pth
                pth_files = [file for file in files if file.endswith(".pth")]
                return pth_files
            except FileNotFoundError:
                return [f"Error: The directory '{checkpoint_dir}' does not exist."]

        self._checkpointer = TorchtuneCheckpointer(
            model_id=self.model_id,
            training_algorithm="sft",
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_files=get_checkpoint_files(self.checkpoint_dir),
            output_dir=self._output_dir,
            model_type=await utils.get_checkpointer_model_type(self.model_id),
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()
        return checkpoint_dict

    async def setup(self) -> None:
        checkpoint_dict = await self.load_checkpoint()

        self._model = await self._setup_model(
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            base_model_state_dict=checkpoint_dict[training.MODEL_KEY],
            lora_weights_state_dict=None,
        )
        log.info(f"Model is initialized with precision {self._dtype}.")

        self._tokenizer = await self._setup_tokenizer()
        log.info("Tokenizer is initialized.")

        assert isinstance(self.training_config.optimizer_config, OptimizerConfig), "OptimizerConfig must be initialized"
        self._optimizer = await self._setup_optimizer(optimizer_config=self.training_config.optimizer_config)
        log.info("Optimizer is initialized.")

        self._loss_fn = CEWithChunkedOutputLoss()
        self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)  # type: ignore[operator]
        log.info("Loss is initialized.")

        assert isinstance(self.training_config.data_config, DataConfig), "DataConfig must be initialized"

        self._training_sampler, self._training_dataloader = await self._setup_data(
            dataset_id=self.training_config.data_config.dataset_id,
            tokenizer=self._tokenizer,
            shuffle=self._shuffle,
            batch_size=self._batch_size,
        )

        if self.training_config.data_config.validation_dataset_id:
            _, self._validation_dataloader = await self._setup_data(
                dataset_id=self.training_config.data_config.validation_dataset_id,
                tokenizer=self._tokenizer,
                shuffle=False,
                batch_size=self._batch_size,
            )

        log.info("Dataset and Sampler are initialized.")

        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader and the max_steps_per_epoch param set by the user and is used
        # for logging and tracking training state. This should be computed after the dataloader
        # has been setup
        self._steps_per_epoch = len(self._training_dataloader) // self._gradient_accumulation_steps
        if self.max_steps_per_epoch is not None and self.max_steps_per_epoch < self._steps_per_epoch:
            self._steps_per_epoch = self.max_steps_per_epoch
            self.global_step = self.epochs_run * self._steps_per_epoch

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = await self._setup_lr_scheduler(
            num_warmup_steps=self.training_config.optimizer_config.num_warmup_steps,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )
        log.info("Learning rate scheduler is initialized.")

        # Used to ignore labels for loss computation
        self.ignore_labels_cache = torch.full((self._batch_size, 1), self._loss_fn.ignore_index, device=self._device)

    def _log_memory_stats(self):
        # torchtune raises: "Logging memory stats is not supported on CPU devices"; do nothing
        if self._device.type == "cpu":
            return
        memory_stats = training.get_memory_stats(device=self._device)
        training.log_memory_stats(memory_stats)

    async def _setup_model(
        self,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        base_model_state_dict: dict[str, Any],
        lora_weights_state_dict: dict[str, Any] | None = None,
    ) -> nn.Module:
        self._lora_rank = self.algorithm_config.rank
        self._lora_alpha = self.algorithm_config.alpha
        self._lora_attn_modules = list(self.algorithm_config.lora_attn_modules)
        self._apply_lora_to_mlp = self.algorithm_config.apply_lora_to_mlp
        self._apply_lora_to_output = self.algorithm_config.apply_lora_to_output
        self._use_dora = self.algorithm_config.use_dora or False

        with training.set_default_dtype(self._dtype), self._device:
            model_type = await utils.get_model_definition(self.model_id)
            model = model_type(
                lora_attn_modules=self._lora_attn_modules,
                apply_lora_to_mlp=self._apply_lora_to_mlp,
                apply_lora_to_output=self._apply_lora_to_output,
                lora_rank=self._lora_rank,
                lora_alpha=self._lora_alpha,
                quantize_base=False,
                use_dora=self._use_dora,
            )

        self.adapter_params = get_adapter_params(model)
        self._is_dora = any("magnitude" in k for k in self.adapter_params.keys())

        set_trainable_params(model, self.adapter_params)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(model, auto_wrap_policy={modules.TransformerSelfAttentionLayer})

        base_missing, base_unexpected = model.load_state_dict(base_model_state_dict, strict=False)

        # This is for any adapters that need to be initialized after base weights
        # have been loaded (e.g. DoRA).
        if self._is_dora:
            for m in model.modules():
                if hasattr(m, "initialize_dora_magnitude"):
                    m.initialize_dora_magnitude()  # type: ignore[operator]
        if lora_weights_state_dict:
            lora_missing, lora_unexpected = model.load_state_dict(lora_weights_state_dict, strict=False)
        else:
            lora_missing, lora_unexpected = None, None
        validate_missing_and_unexpected_for_lora(
            lora_attn_modules=self._lora_attn_modules,
            apply_lora_to_mlp=self._apply_lora_to_mlp,
            apply_lora_to_output=self._apply_lora_to_output,
            base_missing=base_missing,
            base_unexpected=base_unexpected,
            lora_missing=lora_missing,
            lora_unexpected=lora_unexpected,
        )

        # Validate model adapter params were loaded in with the expected dtype
        training.validate_expected_param_dtype(self.adapter_params.items(), dtype=self._dtype)

        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(model, enable_activation_offloading)

        self._log_memory_stats()

        return model

    async def _setup_tokenizer(
        self,
    ) -> Llama3Tokenizer:
        tokenizer_path = self.checkpoint_dir + "/tokenizer.model"
        tokenizer_type = await utils.get_tokenizer_type(self.model_id)
        return tokenizer_type(path=tokenizer_path)

    async def _setup_optimizer(self, optimizer_config: OptimizerConfig) -> Optimizer:
        optimizer = torch.optim.AdamW(
            params=self._model.parameters(),
            lr=optimizer_config.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
        )
        return optimizer

    async def _setup_data(
        self,
        dataset_id: str,
        tokenizer: Llama3Tokenizer,
        shuffle: bool,
        batch_size: int,
    ) -> tuple[DistributedSampler, DataLoader]:
        async def fetch_rows(dataset_id: str):
            return await self.datasetio_api.iterrows(
                dataset_id=dataset_id,
                limit=-1,
            )

        all_rows = await fetch_rows(dataset_id)
        rows = all_rows.data

        # TODO (xiyan): validate dataset schema
        # dataset_def = await self.datasets_api.get_dataset(dataset_id=dataset_id)

        data_transform = await utils.get_data_transform(self._data_format)
        ds = SFTDataset(
            rows,
            message_transform=data_transform(train_on_input=self._train_on_input),
            model_transform=tokenizer,
            dataset_type=self._data_format.value,
        )

        sampler: DistributedSampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=batch_size,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
            collate_fn=(
                partial(
                    padded_collate_sft,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
            ),
        )

        return sampler, dataloader

    async def _setup_lr_scheduler(
        self,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = get_cosine_schedule_with_warmup(
            self._optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
        return lr_scheduler  # type: ignore[no-any-return]

    async def save_checkpoint(self, epoch: int) -> str:
        ckpt_dict = {}

        adapter_state_dict = get_adapter_state_dict(self._model.state_dict())
        ckpt_dict.update({training.ADAPTER_KEY: adapter_state_dict})

        # Construct the full state dict with LoRA weights merged into base LLM weights
        # Move to CPU to avoid a copy on GPU
        state_dict = {k: v.cpu() for k, v in self._model.state_dict().items()}

        merged_state_dict = get_merged_lora_ckpt(
            state_dict,
            rank=self._lora_rank,
            alpha=self._lora_alpha,
        )

        ckpt_dict.update({training.MODEL_KEY: merged_state_dict})

        adapter_config = {
            "r": self._lora_rank,
            "lora_alpha": self._lora_alpha,
            "target_modules": get_lora_module_names(
                self._lora_attn_modules,
                self._apply_lora_to_mlp,
                self._apply_lora_to_output,
            ),
            "peft_type": "LORA",
        }
        ckpt_dict.update({training.ADAPTER_CONFIG: adapter_config})

        return self._checkpointer.save_checkpoint(
            ckpt_dict,
            epoch=epoch,
            checkpoint_format=self._checkpoint_format,
        )

    async def _loss_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Shape [b, s], needed for the loss not the model
        labels = batch.pop("labels")
        # run model
        with self.activations_handling_ctx:
            logits = self._model(**batch)

        # Shift labels to compute loss
        # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
        # But this way we dont need to slice the logits. We just add an ignore index to labels.
        labels = torch.hstack((labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]]))
        if not isinstance(logits, list):
            labels = labels.reshape(-1)
            logits = logits.reshape(-1, logits.size(-1))

        loss = self._loss_fn(logits, labels)

        # free logits otherwise it peaks backward memory
        del logits

        return loss  # type: ignore[no-any-return]

    async def train(self) -> tuple[dict[str, Any], list[Checkpoint]]:
        """
        The core training loop.
        """
        assert isinstance(self.training_config.data_config, DataConfig), "DataConfig must be initialized"
        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss: float = 0.0
        num_tokens = 0

        # training artifacts
        checkpoints = []
        memory_stats: dict[str, Any] = {}

        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            # Update the sampler to ensure data is correctly shuffled across epochs
            # in case shuffle is True
            metric_logger = DiskLogger(log_dir=self._output_dir + f"/{self.model_id}-sft-{curr_epoch}/log")
            self._training_sampler.set_epoch(curr_epoch)
            loss_to_log = 0.0

            pbar = tqdm(total=self._steps_per_epoch)
            for idx, batch in enumerate(self._training_dataloader):
                if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps) == self.max_steps_per_epoch
                ):
                    break

                torchtune_utils.batch_to_device(batch, self._device)

                # Calculate the number of unmasked tokens in the current batch
                # and increment the total number of tokens seen in the step
                current_num_tokens = (batch["labels"] != self._loss_fn.ignore_index).sum()
                num_tokens += current_num_tokens

                # Loss is normalized by default so we multiply by the number of tokens
                # This way we can normalize by the total number of tokens if we're accumulating gradients
                current_loss = await self._loss_step(batch) * current_num_tokens
                running_loss += current_loss.detach().item()
                current_loss.backward()

                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    training.scale_grads(self._model, 1 / num_tokens)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(),
                        max_norm=float(self._clip_grad_norm),
                    )
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    self._lr_scheduler.step()
                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    loss_to_log = running_loss / num_tokens

                    pbar.update(1)
                    pbar.set_description(f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}")

                    time_per_step = time.perf_counter() - t0
                    log_dict = {
                        "loss": loss_to_log,
                        "lr": self._optimizer.param_groups[0]["lr"],
                        "tokens_per_second_per_gpu": num_tokens / time_per_step,
                    }

                    self._log_memory_stats()

                    if self._clip_grad_norm is not None:
                        log_dict.update({"grad_norm": grad_norm})

                    metric_logger.log_dict(
                        log_dict,
                        step=self.global_step,
                    )

                    # Reset running stats for the next step
                    running_loss = 0.0
                    num_tokens = 0
                    t0 = time.perf_counter()

            self.epochs_run += 1
            log.info("Starting checkpoint save...")
            checkpoint_path = await self.save_checkpoint(epoch=curr_epoch)
            checkpoint = Checkpoint(
                identifier=f"{self.model_id}-sft-{curr_epoch}",
                created_at=datetime.now(UTC),
                epoch=curr_epoch,
                post_training_job_id=self.job_uuid,
                path=checkpoint_path,
            )
            if self.training_config.data_config.validation_dataset_id:
                validation_loss, perplexity = await self.validation()
                training_metrics = PostTrainingMetric(
                    epoch=curr_epoch,
                    train_loss=loss_to_log,
                    validation_loss=validation_loss,
                    perplexity=perplexity,
                )
                checkpoint.training_metrics = training_metrics
            checkpoints.append(checkpoint)

        # clean up the memory after training finishes
        evacuate_model_from_device(self._model, self._device.type)

        return (memory_stats, checkpoints)

    async def validation(self) -> tuple[float, float]:
        total_loss = 0.0
        total_tokens = 0
        log.info("Starting validation...")
        pbar = tqdm(total=len(self._validation_dataloader))
        for idx, batch in enumerate(self._validation_dataloader):
            if idx == self.max_validation_steps:
                break
            torchtune_utils.batch_to_device(batch, self._device)

            # Calculate the number of unmasked tokens in the current batch
            # and increment the total number of tokens seen in the step
            num_tokens = (batch["labels"] != self._loss_fn.ignore_index).sum()

            # Loss is normalized by default so we multiply by the number of tokens
            # This way we can normalize by the total number of tokens if we're accumulating gradients
            loss = await self._loss_step(batch) * num_tokens

            total_loss += loss
            total_tokens += num_tokens

            pbar.update(1)
            pbar.set_description(f"validation step: {idx}")

        mean_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(mean_loss))

        return mean_loss, perplexity.item()
