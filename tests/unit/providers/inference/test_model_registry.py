import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from llama_stack.core.request_headers import request_provider_data_context
from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin
from llama_stack.providers.utils.inference.model_registry import ProviderModelEntry, ModelType

class TestConfig(BaseModel):
    api_key: str | None = Field(default=None)

class TestProviderDataValidator(BaseModel):
    test_api_key: str | None = Field(default=None)

MODEL_ENTRIES_WITHOUT_ALIASES = [
    ProviderModelEntry(model_type=ModelType.llm, provider_model_id="test-llm-model", aliases=[]),
    ProviderModelEntry(model_type=ModelType.embedding, provider_model_id="test-text-embedding-model", aliases=[], metadata={"embedding_dimension": 1536, "context_length": 8192}),
]

class TestLiteLLMAdapterWithModelEntries(LiteLLMOpenAIMixin):
    def __init__(self, config: TestConfig):
        super().__init__(
            model_entries=MODEL_ENTRIES_WITHOUT_ALIASES,
            litellm_provider_name="test",
            api_key_from_config=config.api_key,
            provider_data_api_key_field="test_api_key",
            openai_compat_api_base=None,
        )

@pytest.fixture
def adapter_with_model_entries():
    """Fixture to create adapter with API key in config"""
    config = TestConfig()
    adapter = TestLiteLLMAdapterWithModelEntries(config)
    adapter.__provider_id__ = "test-provider"

    return adapter

async def test_model_types_are_correct(adapter_with_model_entries):
    """Test that model types are correct"""
    model_entries = adapter_with_model_entries.model_entries
    llm_model_entries = [m for m in model_entries if m.model_type == ModelType.llm]
    assert len(llm_model_entries) == 1

    embedding_model_entries = [m for m in model_entries if m.model_type == ModelType.embedding]
    assert len(embedding_model_entries) == 1

    models = await adapter_with_model_entries.list_models()
    llm_models = [m for m in models if m.model_type == ModelType.llm]
    assert len(llm_models) == len(llm_model_entries)

    embedding_models = [m for m in models if m.model_type == ModelType.embedding]
    assert len(embedding_models) == len(embedding_model_entries)

def test_embedding_metadata_is_required():
    with pytest.raises(ValueError):
        entry1 = ProviderModelEntry(
            model_type=ModelType.embedding,
            provider_model_id="test-text-embedding-model",
            aliases=[],
            metadata={}
        )
    
    entry2 = ProviderModelEntry(
        model_type=ModelType.embedding,
        provider_model_id="test-text-embedding-model",
        aliases=[],
        metadata={"embedding_dimension": 1536}
    )
    assert entry2.metadata["embedding_dimension"] == 1536
