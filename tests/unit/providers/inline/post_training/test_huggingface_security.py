# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.inline.post_training.huggingface.config import (
    HuggingFacePostTrainingConfig,
)


class TestHuggingFacePostTrainingSecurityConfig:
    """Test HuggingFace post-training security configuration"""

    def test_default_config(self):
        """Default config should have minimal trusted prefixes"""
        config = HuggingFacePostTrainingConfig(dpo_output_dir="/tmp/output")
        assert "nomic-ai/" in config.trusted_model_prefixes

    def test_model_specific_config_trusted_model(self):
        """Trusted models should get trust_remote_code=True"""
        config = HuggingFacePostTrainingConfig(dpo_output_dir="/tmp/output")
        model_config = config.get_model_specific_config("nomic-ai/nomic-embed-text-v1.5")

        assert "trust_remote_code" in model_config
        assert model_config["trust_remote_code"] is True
        assert "attn_implementation" in model_config
        assert model_config["attn_implementation"] == "sdpa"

    def test_model_specific_config_untrusted_model(self):
        """Untrusted models should get trust_remote_code=False"""
        config = HuggingFacePostTrainingConfig(dpo_output_dir="/tmp/output")
        model_config = config.get_model_specific_config("meta-llama/Llama-3.3-70B-Instruct")

        assert "trust_remote_code" in model_config
        assert model_config["trust_remote_code"] is False
        assert "attn_implementation" in model_config
        assert model_config["attn_implementation"] == "sdpa"

    def test_model_specific_config_preserves_base_config(self):
        """get_model_specific_config should preserve base model_specific_config"""
        config = HuggingFacePostTrainingConfig(dpo_output_dir="/tmp/output")
        config.model_specific_config = {
            "attn_implementation": "custom",
            "custom_key": "custom_value",
        }

        model_config = config.get_model_specific_config("meta-llama/Llama-3.3-70B")

        assert model_config["attn_implementation"] == "custom"
        assert model_config["custom_key"] == "custom_value"
        assert model_config["trust_remote_code"] is False

    def test_model_specific_config_does_not_mutate_original(self):
        """get_model_specific_config should not mutate the original config"""
        config = HuggingFacePostTrainingConfig(dpo_output_dir="/tmp/output")
        original_config = config.model_specific_config.copy()

        # Get config for a model
        config.get_model_specific_config("meta-llama/Llama-3.3-70B")

        # Original should be unchanged
        assert config.model_specific_config == original_config
        assert "trust_remote_code" not in config.model_specific_config

    def test_custom_trusted_prefixes(self):
        """Custom trusted prefixes should work"""
        config = HuggingFacePostTrainingConfig(
            dpo_output_dir="/tmp/output", trusted_model_prefixes=["meta-llama/", "mistralai/"]
        )

        # Trusted models
        assert config.get_model_specific_config("meta-llama/Llama-3.3-70B")["trust_remote_code"] is True
        assert config.get_model_specific_config("mistralai/Mistral-7B-v0.1")["trust_remote_code"] is True

        # Untrusted models
        assert config.get_model_specific_config("nomic-ai/nomic-embed-text-v1.5")["trust_remote_code"] is False
        assert config.get_model_specific_config("random/model")["trust_remote_code"] is False

    def test_empty_trusted_prefixes(self):
        """Empty trusted prefixes should deny all models"""
        config = HuggingFacePostTrainingConfig(dpo_output_dir="/tmp/output", trusted_model_prefixes=[])

        assert config.get_model_specific_config("nomic-ai/nomic-embed-text-v1.5")["trust_remote_code"] is False
        assert config.get_model_specific_config("meta-llama/Llama-3.3-70B")["trust_remote_code"] is False
