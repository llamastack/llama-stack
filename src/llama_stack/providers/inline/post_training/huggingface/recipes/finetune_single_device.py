# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import gc
import json
import multiprocessing
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.post_training import (
    Checkpoint,
    DataConfig,
    LoraFinetuningConfig,
    TrainingConfig,
)
from llama_stack.log import get_logger
from llama_stack.providers.inline.post_training.common.utils import evacuate_model_from_device

from ..config import HuggingFacePostTrainingConfig
from ..utils import (
    HFAutoModel,
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


class HFFinetuningSingleDevice:
    def __init__(
        self,
        job_uuid: str,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
    ):
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api
        self.job_uuid = job_uuid

    def validate_dataset_format(self, rows: list[dict]) -> bool:
        """Validate that the dataset has the required fields."""
        required_fields = ["input_query", "expected_answer", "chat_completion_input"]
        return all(field in row for row in rows for field in required_fields)

    def _process_instruct_format(self, row: dict) -> tuple[str | None, str | None]:
        """Process a row in instruct format."""
        if "chat_completion_input" in row and "expected_answer" in row:
            try:
                messages = json.loads(row["chat_completion_input"])
                if not isinstance(messages, list) or len(messages) != 1:
                    logger.warning(f"Invalid chat_completion_input format: {row['chat_completion_input']}")
                    return None, None
                if "content" not in messages[0]:
                    logger.warning(f"Message missing content: {messages[0]}")
                    return None, None
                return messages[0]["content"], row["expected_answer"]
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse chat_completion_input: {row['chat_completion_input']}")
                return None, None
        return None, None

    def _process_dialog_format(self, row: dict) -> tuple[str | None, str | None]:
        """Process a row in dialog format."""
        if "dialog" in row:
            try:
                dialog = json.loads(row["dialog"])
                if not isinstance(dialog, list) or len(dialog) < 2:
                    logger.warning(f"Dialog must have at least 2 messages: {row['dialog']}")
                    return None, None
                if dialog[0].get("role") != "user":
                    logger.warning(f"First message must be from user: {dialog[0]}")
                    return None, None
                if not any(msg.get("role") == "assistant" for msg in dialog):
                    logger.warning("Dialog must have at least one assistant message")
                    return None, None

                # Convert to human/gpt format
                role_map = {"user": "human", "assistant": "gpt"}
                conversations = []
                for msg in dialog:
                    if "role" not in msg or "content" not in msg:
                        logger.warning(f"Message missing role or content: {msg}")
                        continue
                    conversations.append({"from": role_map[msg["role"]], "value": msg["content"]})

                # Format as a single conversation
                return conversations[0]["value"], conversations[1]["value"]
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse dialog: {row['dialog']}")
                return None, None
        return None, None

    def _process_fallback_format(self, row: dict) -> tuple[str | None, str | None]:
        """Process a row using fallback formats."""
        if "input" in row and "output" in row:
            return row["input"], row["output"]
        elif "prompt" in row and "completion" in row:
            return row["prompt"], row["completion"]
        elif "question" in row and "answer" in row:
            return row["question"], row["answer"]
        return None, None

    def _format_text(self, input_text: str, output_text: str, provider_config: HuggingFacePostTrainingConfig) -> str:
        """Format input and output text based on model requirements."""
        if hasattr(provider_config, "chat_template"):
            return provider_config.chat_template.format(input=input_text, output=output_text)
        return f"{input_text}\n{output_text}"

    def _create_dataset(
        self, rows: list[dict], config: TrainingConfig, provider_config: HuggingFacePostTrainingConfig
    ) -> Dataset:
        """Create and preprocess the dataset."""
        formatted_rows = []
        for row in rows:
            input_text = None
            output_text = None

            # Process based on format
            assert isinstance(config.data_config, DataConfig), "DataConfig must be initialized"
            if config.data_config.data_format.value == "instruct":
                input_text, output_text = self._process_instruct_format(row)
            elif config.data_config.data_format.value == "dialog":
                input_text, output_text = self._process_dialog_format(row)
            else:
                input_text, output_text = self._process_fallback_format(row)

            if input_text and output_text:
                formatted_text = self._format_text(input_text, output_text, provider_config)
                formatted_rows.append({"text": formatted_text})

        if not formatted_rows:
            assert isinstance(config.data_config, DataConfig), "DataConfig must be initialized"
            raise ValueError(
                f"No valid input/output pairs found in the dataset for format: {config.data_config.data_format.value}"
            )

        return Dataset.from_list(formatted_rows)

    def _preprocess_dataset(
        self, ds: Dataset, tokenizer: AutoTokenizer, provider_config: HuggingFacePostTrainingConfig
    ) -> Dataset:
        """Preprocess the dataset with tokenizer."""

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding=True,
                truncation=True,
                max_length=provider_config.max_seq_length,
                return_tensors=None,
            )

        return ds.map(
            tokenize_function,
            batched=True,
            remove_columns=ds.column_names,
        )

    def _run_training_sync(
        self,
        model: str,
        provider_config: dict[str, Any],
        peft_config: LoraConfig | None,
        config: dict[str, Any],
        output_dir_path: Path | None,
    ) -> None:
        """Synchronous wrapper for running training process.
        This method serves as a bridge between the multiprocessing Process and the async training function.
        It creates a new event loop to run the async training process.
        Args:
            model: The model identifier to load
            dataset_id: ID of the dataset to use for training
            provider_config: Configuration specific to the HuggingFace provider
            peft_config: Optional LoRA configuration
            config: General training configuration
            output_dir_path: Optional path to save the model
        """
        import asyncio

        logger.info("Starting training process with async wrapper")
        asyncio.run(
            self._run_training(
                model=model,
                provider_config=provider_config,
                peft_config=peft_config,
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
        """Load and prepare the dataset for training.
        Args:
            model: The model identifier to load
            config: Training configuration
            provider_config: Provider-specific configuration
        Returns:
            tuple: (train_dataset, eval_dataset, tokenizer)
        """
        # Validate data config
        if not config.data_config:
            raise ValueError("DataConfig is required for training")

        # Load dataset
        logger.info(f"Loading dataset: {config.data_config.dataset_id}")
        rows = await load_rows_from_dataset(self.datasetio_api, config.data_config.dataset_id)
        if not self.validate_dataset_format(rows):
            raise ValueError("Dataset is missing required fields: input_query, expected_answer, chat_completion_input")
        logger.info(f"Loaded {len(rows)} rows from dataset")

        # Initialize tokenizer
        logger.info(f"Initializing tokenizer for model: {model}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model, **provider_config.model_specific_config)

            # Set pad token to eos token if not present
            # This is common for models that don't have a dedicated pad token
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token

            # Set padding side to right for causal language modeling
            # This ensures that padding tokens don't interfere with the model's ability
            # to predict the next token in the sequence
            tokenizer.padding_side = "right"

            # Set truncation side to right to keep the beginning of the sequence
            # This is important for maintaining context and instruction format
            tokenizer.truncation_side = "right"

            # Set model max length to match provider config
            # This ensures consistent sequence lengths across the training process
            tokenizer.model_max_length = provider_config.max_seq_length

            logger.info("Tokenizer initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tokenizer: {str(e)}") from e

        # Create and preprocess dataset
        logger.info("Creating and preprocessing dataset")
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
        device: torch.device,
        output_dir_path: Path | None,
        steps_per_epoch: int,
    ) -> SFTConfig:
        """Setup training arguments.
        Args:
            config: Training configuration
            provider_config: Provider-specific configuration
            device: The device to train on
            output_dir_path: Optional path to save the model
            steps_per_epoch: Number of steps per epoch
        Returns:
            Configured SFTConfig object
        """
        logger.info("Configuring training arguments")
        lr = 2e-5
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

        return SFTConfig(
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
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_checkpointing=provider_config.gradient_checkpointing,
            learning_rate=lr,
            warmup_ratio=provider_config.warmup_ratio,
            weight_decay=provider_config.weight_decay,
            remove_unused_columns=False,
            dataloader_pin_memory=provider_config.dataloader_pin_memory,
            dataloader_num_workers=provider_config.dataloader_num_workers,
            dataset_text_field="text",
            packing=False,
            load_best_model_at_end=True if output_dir_path else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=step_info["logging_steps"],
        )

    def save_model(
        self,
        model_obj: HFAutoModel,
        trainer: SFTTrainer,
        peft_config: LoraConfig | None,
        output_dir_path: Path,
    ) -> None:
        """Save the trained model.
        Args:
            model_obj: The model to save
            trainer: The trainer instance
            peft_config: Optional LoRA configuration
            output_dir_path: Path to save the model
        """
        from typing import cast

        logger.info("Saving final model")
        model_obj.config.use_cache = True

        if peft_config:
            logger.info("Merging LoRA weights with base model")
            # TRL's merge_and_unload returns a HuggingFace model
            # Both cast() and type: ignore are needed here:
            # - cast() tells mypy the return type is HFAutoModel for downstream code
            # - type: ignore suppresses errors on the merge_and_unload() call itself,
            #   which mypy can't type-check due to TRL library's incomplete type stubs
            model_obj = cast(HFAutoModel, trainer.model.merge_and_unload())  # type: ignore[union-attr,operator]
        else:
            # trainer.model is the trained HuggingFace model
            model_obj = cast(HFAutoModel, trainer.model)

        save_path = output_dir_path / "merged_model"
        logger.info(f"Saving model to {save_path}")
        model_obj.save_pretrained(save_path)

    async def _run_training(
        self,
        model: str,
        provider_config: dict[str, Any],
        peft_config: LoraConfig | None,
        config: dict[str, Any],
        output_dir_path: Path | None,
    ) -> None:
        """Run the training process with signal handling."""

        # Setup environment variables
        setup_environment()

        # Setup signal handlers
        setup_signal_handlers()

        # Convert config dicts back to objects
        logger.info("Initializing configuration objects")
        provider_config_obj = HuggingFacePostTrainingConfig(**provider_config)
        config_obj = TrainingConfig(**config)

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
            device,
            output_dir_path,
            steps_per_epoch,
        )

        # Load model
        model_obj = load_model(model, device, provider_config_obj)

        # Initialize trainer
        logger.info("Initializing SFTTrainer")
        trainer = SFTTrainer(
            model=model_obj,  # type: ignore[arg-type]
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            args=training_args,
        )

        try:
            # Train
            logger.info("Starting training")
            trainer.train()
            logger.info("Training completed successfully")

            # Save final model if output directory is provided
            if output_dir_path:
                self.save_model(model_obj, trainer, peft_config, output_dir_path)

        finally:
            # Clean up resources
            logger.info("Cleaning up resources")
            if hasattr(trainer, "model"):
                evacuate_model_from_device(trainer.model, device.type)
            del trainer
            gc.collect()
            logger.info("Cleanup completed")

    async def train(
        self,
        model: str,
        output_dir: str | None,
        job_uuid: str,
        lora_config: LoraFinetuningConfig,
        config: TrainingConfig,
        provider_config: HuggingFacePostTrainingConfig,
    ) -> tuple[dict[str, Any], list[Checkpoint] | None]:
        """Train a model using HuggingFace's SFTTrainer"""
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

        # Configure LoRA
        peft_config = None
        if lora_config:
            peft_config = LoraConfig(
                lora_alpha=lora_config.alpha,
                lora_dropout=0.1,
                r=lora_config.rank,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_config.lora_attn_modules,
            )

        # Validate data config
        if not config.data_config:
            raise ValueError("DataConfig is required for training")

        # Train in a separate process
        logger.info("Starting training in separate process")
        try:
            # Setup multiprocessing for device
            if device.type in ["cuda", "mps"]:
                multiprocessing.set_start_method("spawn", force=True)

            process = multiprocessing.Process(
                target=self._run_training_sync,
                kwargs={
                    "model": model,
                    "provider_config": provider_config.model_dump(),
                    "peft_config": peft_config,
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
                raise RuntimeError(f"Training failed with exit code {process.exitcode}")

            memory_stats["after_training"] = get_memory_stats(device)

            checkpoints = []
            if output_dir_path:
                checkpoints = create_checkpoints(output_dir_path, job_uuid, model, config, "merged_model")

            return memory_stats, checkpoints if checkpoints else None
        finally:
            memory_stats["final"] = get_memory_stats(device)
            gc.collect()
