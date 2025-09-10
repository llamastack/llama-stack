# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import sys
import time
import uuid

import pytest

from llama_stack.apis.post_training import (
    DataConfig,
    DatasetFormat,
    DPOAlignmentConfig,
    DPOLossType,
    LoraFinetuningConfig,
    TrainingConfig,
)
from llama_stack.log import get_logger

# Configure logging
logger = get_logger(name=__name__, category="post_training")


skip_because_resource_intensive = pytest.mark.skip(
    reason="""
       Post training tests are extremely resource intensive. They download large models and partly as a result,
       are very slow to run. We cannot run them on every single PR update. CI should be considered
       a scarce resource and properly utilitized.
    """
)


@pytest.fixture(autouse=True)
def capture_output(capsys):
    """Fixture to capture and display output during test execution."""
    yield
    captured = capsys.readouterr()
    if captured.out:
        print("\nCaptured stdout:", captured.out)
    if captured.err:
        print("\nCaptured stderr:", captured.err)


# Force flush stdout to see prints immediately
sys.stdout.reconfigure(line_buffering=True)

# How to run this test:
#
# LLAMA_STACK_CONFIG=ci-tests uv run --dev pytest tests/integration/post_training/test_post_training.py
#


# SFT test
class TestPostTraining:
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "purpose, source",
        [
            (
                "post-training/messages",
                {
                    "type": "uri",
                    "uri": "huggingface://datasets/llamastack/simpleqa?split=train",
                },
            ),
        ],
    )
    @pytest.mark.timeout(360)  # 6 minutes timeout
    def test_supervised_fine_tune(self, llama_stack_client, purpose, source):
        logger.info("Starting supervised fine-tuning test")

        # register dataset to train
        dataset = llama_stack_client.datasets.register(
            purpose=purpose,
            source=source,
        )
        logger.info(f"Registered dataset with ID: {dataset.identifier}")

        algorithm_config = LoraFinetuningConfig(
            type="LoRA",
            lora_attn_modules=["q_proj", "v_proj", "output_proj"],
            apply_lora_to_mlp=True,
            apply_lora_to_output=False,
            rank=8,
            alpha=16,
        )

        data_config = DataConfig(
            dataset_id=dataset.identifier,
            batch_size=1,
            shuffle=False,
            data_format=DatasetFormat.instruct,
        )

        # setup training config with minimal settings
        training_config = TrainingConfig(
            n_epochs=1,
            data_config=data_config,
            max_steps_per_epoch=1,
            gradient_accumulation_steps=1,
        )

        job_uuid = f"test-job{uuid.uuid4()}"
        logger.info(f"Starting training job with UUID: {job_uuid}")

        # train with HF trl SFTTrainer as the default
        _ = llama_stack_client.post_training.supervised_fine_tune(
            job_uuid=job_uuid,
            model="HuggingFaceTB/SmolLM2-135M-Instruct",  # smaller model that supports the current sft recipe
            algorithm_config=algorithm_config,
            training_config=training_config,
            hyperparam_search_config={},
            logger_config={},
            checkpoint_dir=None,
        )

        while True:
            status = llama_stack_client.post_training.job.status(job_uuid=job_uuid)
            if not status:
                logger.error("Job not found")
                break

            logger.info(f"Current status: {status}")
            assert status.status in ["scheduled", "in_progress", "completed"]
            if status.status == "completed":
                break

            logger.info("Waiting for job to complete...")
            time.sleep(10)  # Increased sleep time to reduce polling frequency

        artifacts = llama_stack_client.post_training.job.artifacts(job_uuid=job_uuid)
        logger.info(f"Job artifacts: {artifacts}")

        logger.info(f"Registered dataset with ID: {dataset.identifier}")

    # TODO: Fix these tests to properly represent the Jobs API in training
    #
    # async def test_get_training_jobs(self, post_training_stack):
    #     post_training_impl = post_training_stack
    #     jobs_list = await post_training_impl.get_training_jobs()
    #     assert isinstance(jobs_list, list)
    #     assert jobs_list[0].job_uuid == "1234"

    #
    # async def test_get_training_job_status(self, post_training_stack):
    #     post_training_impl = post_training_stack
    #     job_status = await post_training_impl.get_training_job_status("1234")
    #     assert isinstance(job_status, PostTrainingJobStatusResponse)
    #     assert job_status.job_uuid == "1234"
    #     assert job_status.status == JobStatus.completed
    #     assert isinstance(job_status.checkpoints[0], Checkpoint)

    #
    # async def test_get_training_job_artifacts(self, post_training_stack):
    #     post_training_impl = post_training_stack
    #     job_artifacts = await post_training_impl.get_training_job_artifacts("1234")
    #     assert isinstance(job_artifacts, PostTrainingJobArtifactsResponse)
    #     assert job_artifacts.job_uuid == "1234"
    #     assert isinstance(job_artifacts.checkpoints[0], Checkpoint)
    #     assert job_artifacts.checkpoints[0].identifier == "instructlab/granite-7b-lab"
    #     assert job_artifacts.checkpoints[0].epoch == 0
    # assert "/.llama/checkpoints/Llama3.2-3B-Instruct-sft-0" in job_artifacts.checkpoints[0].path

    # DPO test
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "purpose, source",
        [
            (
                "post-training/messages",
                {
                    "type": "uri",
                    "uri": "huggingface://datasets/trl-internal-testing/hh-rlhf-helpful-base-trl-style?split=train[:20]",
                },
            ),
        ],
    )
    @pytest.mark.timeout(360)
    def test_preference_optimize(self, llama_stack_client, purpose, source):
        logger.info("Starting DPO preference optimization test")

        # register preference dataset to train
        dataset = llama_stack_client.datasets.register(
            purpose=purpose,
            source=source,
        )
        logger.info(f"Registered preference dataset with ID: {dataset.identifier}")

        # DPO algorithm configuration
        algorithm_config = DPOAlignmentConfig(
            beta=0.1,
            loss_type=DPOLossType.sigmoid,  # Default loss type
        )
        data_config = DataConfig(
            dataset_id=dataset.identifier,
            batch_size=1,
            shuffle=False,
            data_format=DatasetFormat.dialog,  # DPO datasets often use dialog format
        )

        # setup training config with minimal settings for DPO
        training_config = TrainingConfig(
            n_epochs=1,
            data_config=data_config,
            max_steps_per_epoch=1,  # Just 2 steps for quick testing
            gradient_accumulation_steps=1,
        )

        job_uuid = f"test-dpo-job-{uuid.uuid4()}"
        logger.info(f"Starting DPO training job with UUID: {job_uuid}")

        # train with HuggingFace DPO implementation
        _ = llama_stack_client.post_training.preference_optimize(
            job_uuid=job_uuid,
            finetuned_model="distilgpt2",  # Much smaller model for faster CI testing
            algorithm_config=algorithm_config,
            training_config=training_config,
            hyperparam_search_config={},
            logger_config={},
        )

        while True:
            status = llama_stack_client.post_training.job.status(job_uuid=job_uuid)
            if not status:
                logger.error("DPO job not found")
                break

            logger.info(f"Current DPO status: {status}")
            if status.status == "completed":
                break

            logger.info("Waiting for DPO job to complete...")
            time.sleep(10)  # Increased sleep time to reduce polling frequency

        artifacts = llama_stack_client.post_training.job.artifacts(job_uuid=job_uuid)
        logger.info(f"DPO job artifacts: {artifacts}")
