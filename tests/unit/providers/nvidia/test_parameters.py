# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import warnings
from unittest.mock import patch

import pytest

from llama_stack.apis.post_training.post_training import (
    DataConfig,
    DatasetFormat,
    EfficiencyConfig,
    LoraFinetuningConfig,
    OptimizerConfig,
    OptimizerType,
    TrainingConfig,
)
from llama_stack.core.library_client import convert_pydantic_to_json_value
from llama_stack.providers.remote.post_training.nvidia.post_training import (
    NvidiaPostTrainingAdapter,
    NvidiaPostTrainingConfig,
)


class TestNvidiaParameters:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test method."""
        os.environ["NVIDIA_CUSTOMIZER_URL"] = "http://nemo.test"

        config = NvidiaPostTrainingConfig(customizer_url=os.environ["NVIDIA_CUSTOMIZER_URL"], api_key=None)
        self.adapter = NvidiaPostTrainingAdapter(config)

        self.make_request_patcher = patch(
            "llama_stack.providers.remote.post_training.nvidia.post_training.NvidiaPostTrainingAdapter._make_request"
        )
        self.mock_make_request = self.make_request_patcher.start()
        self.mock_make_request.return_value = {
            "id": "job-123",
            "status": "created",
            "created_at": "2025-03-04T13:07:47.543605",
            "updated_at": "2025-03-04T13:07:47.543605",
        }

        yield

        self.make_request_patcher.stop()

    def _assert_request_params(self, expected_json):
        """Helper method to verify parameters in the request JSON."""
        call_args = self.mock_make_request.call_args
        actual_json = call_args[1]["json"]

        for key, value in expected_json.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    assert actual_json[key][nested_key] == nested_value
            else:
                assert actual_json[key] == value

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, run_async):
        self.run_async = run_async

    def test_customizer_parameters_passed(self):
        """Test scenario 1: When an optional parameter is passed and value is correctly set."""
        algorithm_config = LoraFinetuningConfig(
            type="LoRA",
            apply_lora_to_mlp=True,
            apply_lora_to_output=True,
            alpha=16,
            rank=16,
            lora_attn_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        data_config = DataConfig(
            dataset_id="test-dataset", batch_size=16, shuffle=False, data_format=DatasetFormat.instruct
        )
        optimizer_config = OptimizerConfig(
            optimizer_type=OptimizerType.adam,
            lr=0.0002,
            weight_decay=0.01,
            num_warmup_steps=100,
        )
        training_config = TrainingConfig(
            n_epochs=3,
            data_config=data_config,
            optimizer_config=optimizer_config,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            self.run_async(
                self.adapter.supervised_fine_tune(
                    job_uuid="test-job",
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    checkpoint_dir="",
                    algorithm_config=algorithm_config,
                    training_config=convert_pydantic_to_json_value(training_config),
                    logger_config={},
                    hyperparam_search_config={},
                )
            )

            warning_texts = [str(warning.message) for warning in w]

            fields = [
                "apply_lora_to_output",
                "lora_attn_modules",
                "apply_lora_to_mlp",
            ]
            for field in fields:
                assert any(field in text for text in warning_texts)

        self._assert_request_params(
            {
                "hyperparameters": {
                    "lora": {"alpha": 16},
                    "epochs": 3,
                    "learning_rate": 0.0002,
                    "batch_size": 16,
                }
            }
        )

    def test_required_parameters_passed(self):
        """Test scenario 2: When required parameters are passed."""
        required_model = "meta/llama-3.2-1b-instruct@v1.0.0+L40"
        required_dataset_id = "required-dataset"
        required_job_uuid = "required-job"

        algorithm_config = LoraFinetuningConfig(
            type="LoRA",
            apply_lora_to_mlp=True,
            apply_lora_to_output=True,
            alpha=16,
            rank=16,
            lora_attn_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        data_config = DataConfig(
            dataset_id=required_dataset_id, batch_size=8, shuffle=False, data_format=DatasetFormat.instruct
        )

        optimizer_config = OptimizerConfig(
            optimizer_type=OptimizerType.adam,
            lr=0.0001,
            weight_decay=0.01,
            num_warmup_steps=100,
        )

        training_config = TrainingConfig(
            n_epochs=1,
            data_config=data_config,
            optimizer_config=optimizer_config,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            self.run_async(
                self.adapter.supervised_fine_tune(
                    job_uuid=required_job_uuid,
                    model=required_model,
                    checkpoint_dir="",
                    algorithm_config=algorithm_config,
                    training_config=convert_pydantic_to_json_value(training_config),
                    logger_config={},
                    hyperparam_search_config={},
                )
            )

            warning_texts = [str(warning.message) for warning in w]

            fields = [
                "rank",
                "apply_lora_to_output",
                "lora_attn_modules",
                "apply_lora_to_mlp",
            ]
            for field in fields:
                assert any(field in text for text in warning_texts)

        self.mock_make_request.assert_called_once()
        call_args = self.mock_make_request.call_args

        assert call_args[1]["json"]["config"] == required_model
        assert call_args[1]["json"]["dataset"]["name"] == required_dataset_id

    def test_unsupported_parameters_warning(self):
        """Test that warnings are raised for unsupported parameters."""
        data_config = DataConfig(
            dataset_id="test-dataset",
            batch_size=8,
            shuffle=True,
            data_format=DatasetFormat.instruct,
            validation_dataset_id="val-dataset",
        )

        optimizer_config = OptimizerConfig(
            lr=0.0001,
            weight_decay=0.01,
            optimizer_type=OptimizerType.adam,
            num_warmup_steps=100,
        )

        efficiency_config = EfficiencyConfig(enable_activation_checkpointing=True)

        training_config = TrainingConfig(
            n_epochs=1,
            data_config=data_config,
            optimizer_config=optimizer_config,
            efficiency_config=efficiency_config,
            max_steps_per_epoch=1000,
            gradient_accumulation_steps=4,
            max_validation_steps=100,
            dtype="bf16",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            self.run_async(
                self.adapter.supervised_fine_tune(
                    job_uuid="test-job",
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    checkpoint_dir="test-dir",
                    algorithm_config=LoraFinetuningConfig(
                        type="LoRA",
                        apply_lora_to_mlp=True,
                        apply_lora_to_output=True,
                        alpha=16,
                        rank=16,
                        lora_attn_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    ),
                    training_config=convert_pydantic_to_json_value(training_config),
                    logger_config={"test": "value"},
                    hyperparam_search_config={"test": "value"},
                )
            )

            assert len(w) >= 4
            warning_texts = [str(warning.message) for warning in w]

            fields = [
                "checkpoint_dir",
                "hyperparam_search_config",
                "logger_config",
                "TrainingConfig",
                "DataConfig",
                "OptimizerConfig",
                "max_steps_per_epoch",
                "gradient_accumulation_steps",
                "max_validation_steps",
                "dtype",
                "rank",
                "apply_lora_to_output",
                "lora_attn_modules",
                "apply_lora_to_mlp",
            ]
            for field in fields:
                assert any(field in text for text in warning_texts)
