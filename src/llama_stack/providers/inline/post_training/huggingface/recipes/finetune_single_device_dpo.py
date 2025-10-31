# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import gc
import multiprocessing
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
)
from trl import DPOConfig, DPOTrainer

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.post_training import (
    Checkpoint,
    DPOAlignmentConfig,
    TrainingConfig,
)
from llama_stack.log import get_logger
from llama_stack.providers.inline.post_training.common.utils import evacuate_model_from_device

from ..config import HuggingFacePostTrainingConfig
from ..utils import (
    calculate_training_steps,
    create_checkpoints,
    get_memory_stats,
    get_save_strategy,
    load_model,
    load_rows_from_dataset,
    setup_environment,
    setup_signal_handlers,
    setup_torch_device,
    split_dataset,
)

logger = get_logger(name=__name__, category="post_training")


class HFDPOAlignmentSingleDevice:
    def __init__(
        self,
        job_uuid: str,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
    ):
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api
        self.job_uuid = job_uuid

    def validate_dataset_format(self, rows: list[dict]) -> None:
        """Validate that the dataset has the required fields for DPO training."""
        required_fields = ["prompt", "chosen", "rejected"]

        if not rows:
            logger.warning("Dataset is empty")
            raise ValueError("Dataset is empty")

        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                logger.warning(f"Row {i} is not a dictionary")
                raise ValueError(f"Row {i} is not a dictionary")

            for field in required_fields:
                if field not in row:
                    logger.warning(f"Row {i} missing required DPO field: {field}")
                    raise ValueError(f"Row {i} missing required DPO field: {field}")

                # Handle both string and list formats
                if field == "prompt":
                    # Prompt should be a string
                    if not isinstance(row[field], str):
                        logger.warning(f"Row {i} field '{field}' is not a string")
                        raise ValueError(f"Row {i} field '{field}' is not a string")
                    if not row[field].strip():
                        logger.warning(f"Row {i} field '{field}' is empty")
                        raise ValueError(f"Row {i} field '{field}' is empty")
                else:
                    # chosen/rejected can be either strings or lists of messages
                    if isinstance(row[field], str):
                        if not row[field].strip():
                            logger.warning(f"Row {i} field '{field}' is empty")
                            raise ValueError(f"Row {i} field '{field}' is empty")
                    elif isinstance(row[field], list):
                        if not row[field]:
                            logger.warning(f"Row {i} field '{field}' is empty list")
                            raise ValueError(f"Row {i} field '{field}' is empty list")
                    else:
                        logger.warning(f"Row {i} field '{field}' is neither string nor list")
                        raise ValueError(f"Row {i} field '{field}' is neither string nor list")

        logger.info(f"DPO dataset validation passed: {len(rows)} preference examples")

    def _process_dpo_format(self, row: dict) -> tuple[str | None, str | None, str | None]:
        """Process a row in DPO format, handling both string and conversation list formats."""
        if all(field in row for field in ["prompt", "chosen", "rejected"]):
            prompt = row["prompt"]

            # Handle chosen field - convert list to string if needed
            if isinstance(row["chosen"], list):
                # For conversation format, concatenate messages
                chosen = "\n".join(
                    [msg.get("content", "") if isinstance(msg, dict) else str(msg) for msg in row["chosen"]]
                )
            else:
                chosen = row["chosen"]

            # Handle rejected field - convert list to string if needed
            if isinstance(row["rejected"], list):
                # For conversation format, concatenate messages
                rejected = "\n".join(
                    [msg.get("content", "") if isinstance(msg, dict) else str(msg) for msg in row["rejected"]]
                )
            else:
                rejected = row["rejected"]

            return prompt, chosen, rejected
        return None, None, None

    def _format_text_for_dpo(self, prompt: str, response: str, provider_config: HuggingFacePostTrainingConfig) -> str:
        """Format prompt and response text based on model requirements."""
        if hasattr(provider_config, "chat_template") and provider_config.chat_template:
            # Use the chat template, supporting both {prompt}/{response} and {input}/{output}
            template = provider_config.chat_template
            # Try prompt/response first (DPO style)
            if "{prompt}" in template and "{response}" in template:
                return template.format(prompt=prompt, response=response)
            # Fall back to input/output (SFT style)
            elif "{input}" in template and "{output}" in template:
                return template.format(input=prompt, output=response)
            else:
                # If template doesn't have expected placeholders, use default
                return f"{prompt}\n{response}"
        return f"{prompt}\n{response}"

    def _create_dataset(
        self, rows: list[dict], config: TrainingConfig, provider_config: HuggingFacePostTrainingConfig
    ) -> Dataset:
        """Create and preprocess the dataset for DPO."""
        dpo_examples = []
        for row in rows:
            prompt, chosen, rejected = self._process_dpo_format(row)

            if prompt and chosen and rejected:
                # Format the texts
                chosen_formatted = self._format_text_for_dpo(prompt, chosen, provider_config)
                rejected_formatted = self._format_text_for_dpo(prompt, rejected, provider_config)

                dpo_examples.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen_formatted,
                        "rejected": rejected_formatted,
                    }
                )

        if not dpo_examples:
            raise ValueError("No valid preference examples found in dataset")

        logger.info(f"Created DPO dataset with {len(dpo_examples)} preference pairs")
        return Dataset.from_list(dpo_examples)

    def _preprocess_dataset(
        self, ds: Dataset, tokenizer: AutoTokenizer, provider_config: HuggingFacePostTrainingConfig
    ) -> Dataset:
        """Preprocess the dataset with tokenizer for DPO."""
        # DPOTrainer expects raw text, so we don't tokenize here
        # Just return the dataset as is
        return ds

    def _run_training_sync(
        self,
        model: str,
        provider_config: dict[str, Any],
        dpo_config: dict[str, Any],
        config: dict[str, Any],
        output_dir_path: Path | None,
    ) -> None:
        """Synchronous wrapper for running DPO training process."""
        import asyncio

        logger.info("Starting DPO training process with async wrapper")
        asyncio.run(
            self._run_training(
                model=model,
                provider_config=provider_config,
                dpo_config=dpo_config,
                config=config,
                output_dir_path=output_dir_path,
            )
        )

    async def load_dataset(
        self,
        model: str,
        config: TrainingConfig,
        provider_config: HuggingFacePostTrainingConfig,
    ) -> tuple[Dataset, Dataset, AutoTokenizer]:
        """Load and prepare the dataset for DPO training."""
        # Validate data config
        if not config.data_config:
            raise ValueError("DataConfig is required for DPO training")

        # Load dataset
        logger.info(f"Loading dataset: {config.data_config.dataset_id}")
        rows = await load_rows_from_dataset(self.datasetio_api, config.data_config.dataset_id)
        self.validate_dataset_format(rows)
        logger.info(f"Loaded {len(rows)} rows from dataset")

        # Initialize tokenizer
        logger.info(f"Initializing tokenizer for model: {model}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model, **provider_config.model_specific_config)

            # Set pad token to eos token if not present
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token

            # Set padding side to left for DPO
            tokenizer.padding_side = "left"

            # Set truncation side to right to keep the beginning of the sequence
            tokenizer.truncation_side = "right"

            # Set model max length to match provider config
            tokenizer.model_max_length = provider_config.max_seq_length

            logger.info("Tokenizer initialized successfully for DPO")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tokenizer: {str(e)}") from e

        # Create and preprocess dataset
        logger.info("Creating and preprocessing dataset for DPO")
        try:
            ds = self._create_dataset(rows, config, provider_config)
            ds = self._preprocess_dataset(ds, tokenizer, provider_config)
            logger.info(f"Dataset created with {len(ds)} examples")
        except Exception as e:
            raise ValueError(f"Failed to create dataset: {str(e)}") from e

        # Split dataset
        train_dataset, eval_dataset = split_dataset(ds)

        return train_dataset, eval_dataset, tokenizer

    def setup_training_args(
        self,
        config: TrainingConfig,
        provider_config: HuggingFacePostTrainingConfig,
        dpo_config: DPOAlignmentConfig,
        device: torch.device,
        output_dir_path: Path | None,
        steps_per_epoch: int,
    ) -> DPOConfig:
        """Setup DPO training arguments."""
        logger.info("Configuring DPO training arguments")
        lr = 5e-7  # Lower learning rate for DPO
        if config.optimizer_config:
            lr = config.optimizer_config.lr
            logger.info(f"Using custom learning rate: {lr}")

        # Validate data config
        if not config.data_config:
            raise ValueError("DataConfig is required for training")
        data_config = config.data_config

        # Calculate steps and get save strategy
        step_info = calculate_training_steps(steps_per_epoch, config)
        save_strategy, eval_strategy = get_save_strategy(output_dir_path)

        logger.info("DPO training configuration:")
        logger.info(f"- DPO beta: {dpo_config.beta}")
        logger.info(f"- DPO loss type: {provider_config.dpo_loss_type}")

        # Calculate max prompt length as half of max sequence length
        max_prompt_length = provider_config.max_seq_length // 2

        return DPOConfig(
            max_steps=step_info["max_steps"],
            output_dir=str(output_dir_path) if output_dir_path is not None else None,
            num_train_epochs=config.n_epochs,
            per_device_train_batch_size=data_config.batch_size,
            fp16=device.type == "cuda",
            bf16=False,  # Causes CPU issues.
            eval_strategy=eval_strategy,
            use_cpu=True if device.type == "cpu" and not torch.backends.mps.is_available() else False,
            save_strategy=save_strategy,
            report_to="none",
            max_length=provider_config.max_seq_length,
            max_prompt_length=max_prompt_length,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_checkpointing=provider_config.gradient_checkpointing,
            learning_rate=lr,
            warmup_ratio=provider_config.warmup_ratio,
            weight_decay=provider_config.weight_decay,
            remove_unused_columns=False,
            dataloader_pin_memory=provider_config.dataloader_pin_memory,
            dataloader_num_workers=provider_config.dataloader_num_workers,
            load_best_model_at_end=True if output_dir_path else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=step_info["logging_steps"],
            save_total_limit=provider_config.save_total_limit,
            # DPO specific parameters
            beta=dpo_config.beta,
            loss_type=provider_config.dpo_loss_type,  # type: ignore[arg-type]
        )

    def save_model(
        self,
        trainer: DPOTrainer,
        output_dir_path: Path,
    ) -> None:
        """Save the trained DPO model."""
        logger.info("Saving final DPO model")

        save_path = output_dir_path / "dpo_model"
        logger.info(f"Saving model to {save_path}")

        # Save model and tokenizer
        trainer.save_model(str(save_path))

    async def _run_training(
        self,
        model: str,
        provider_config: dict[str, Any],
        dpo_config: dict[str, Any],
        config: dict[str, Any],
        output_dir_path: Path | None,
    ) -> None:
        """Run the DPO training process with signal handling."""

        # Setup environment variables
        setup_environment()

        # Setup signal handlers
        setup_signal_handlers()

        # Convert config dicts back to objects
        logger.info("Initializing configuration objects")
        provider_config_obj = HuggingFacePostTrainingConfig(**provider_config)
        config_obj = TrainingConfig(**config)
        dpo_config_obj = DPOAlignmentConfig(**dpo_config)

        # Initialize and validate device
        device = setup_torch_device(provider_config_obj.device)
        logger.info(f"Using device '{device}'")

        # Load dataset and tokenizer
        train_dataset, eval_dataset, tokenizer = await self.load_dataset(model, config_obj, provider_config_obj)

        # Calculate steps per epoch
        if not config_obj.data_config:
            raise ValueError("DataConfig is required for training")
        steps_per_epoch = len(train_dataset) // config_obj.data_config.batch_size

        # Setup training arguments
        training_args = self.setup_training_args(
            config_obj,
            provider_config_obj,
            dpo_config_obj,
            device,
            output_dir_path,
            steps_per_epoch,
        )

        # Load model and reference model
        model_obj = load_model(model, device, provider_config_obj)
        ref_model = None
        if provider_config_obj.use_reference_model:
            logger.info("Loading separate reference model for DPO")
            ref_model = load_model(model, device, provider_config_obj)
        else:
            logger.info("Using shared reference model for DPO")

        # Initialize DPO trainer
        logger.info("Initializing DPOTrainer")
        # TRL library has incomplete type stubs - use Any to bypass
        from typing import Any, cast

        trainer = DPOTrainer(
            model=cast(Any, model_obj),  # HFAutoModel satisfies PreTrainedModel protocol
            ref_model=cast(Any, ref_model),
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=cast(Any, tokenizer),  # AutoTokenizer satisfies interface
        )

        try:
            # Train
            logger.info("Starting DPO training")
            trainer.train()
            logger.info("DPO training completed successfully")

            # Save final model if output directory is provided
            if output_dir_path:
                logger.info(f"Saving model to output directory: {output_dir_path}")
                self.save_model(trainer, output_dir_path)
                logger.info("Model save completed")

        finally:
            # Clean up resources
            logger.info("Cleaning up resources")
            if hasattr(trainer, "model"):
                evacuate_model_from_device(trainer.model, device.type)
            if ref_model:
                evacuate_model_from_device(ref_model, device.type)
            del trainer
            del ref_model
            gc.collect()
            logger.info("Cleanup completed")
            logger.info("DPO training process finishing successfully")

    async def train(
        self,
        model: str,
        output_dir: str | None,
        job_uuid: str,
        dpo_config: DPOAlignmentConfig,
        config: TrainingConfig,
        provider_config: HuggingFacePostTrainingConfig,
    ) -> tuple[dict[str, Any], list[Checkpoint] | None]:
        """Train a model using HuggingFace's DPOTrainer"""
        # Initialize and validate device
        device = setup_torch_device(provider_config.device)
        logger.info(f"Using device '{device}'")

        output_dir_path = None
        if output_dir:
            output_dir_path = Path(output_dir)

        # Track memory stats
        memory_stats = {
            "initial": get_memory_stats(device),
            "after_training": None,
            "final": None,
        }

        # Validate data config
        if not config.data_config:
            raise ValueError("DataConfig is required for training")

        # Train in a separate process
        logger.info("Starting DPO training in separate process")
        try:
            # Setup multiprocessing for device
            if device.type in ["cuda", "mps"]:
                multiprocessing.set_start_method("spawn", force=True)

            process = multiprocessing.Process(
                target=self._run_training_sync,
                kwargs={
                    "model": model,
                    "provider_config": provider_config.model_dump(),
                    "dpo_config": dpo_config.model_dump(),
                    "config": config.model_dump(),
                    "output_dir_path": output_dir_path,
                },
            )
            process.start()

            # Monitor the process
            while process.is_alive():
                process.join(timeout=1)  # Check every second
                if not process.is_alive():
                    break

            # Get the return code
            if process.exitcode != 0:
                raise RuntimeError(f"DPO training failed with exit code {process.exitcode}")

            memory_stats["after_training"] = get_memory_stats(device)

            checkpoints = []
            if output_dir_path:
                checkpoints = create_checkpoints(output_dir_path, job_uuid, model, config, "dpo_model")

            return memory_stats, checkpoints if checkpoints else None
        finally:
            memory_stats["final"] = get_memory_stats(device)
            gc.collect()
