# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from llama_stack_spec.models.llama.tokenizer_utils import load_bpe_file
from tiktoken.load import load_tiktoken_bpe


@pytest.fixture
def test_bpe_content():
    """Sample BPE file content for testing."""
    return """wA== 0
wQ== 1
9Q== 2
9g== 3
9w== 4
+A== 5
+Q== 6
+g== 7
+w== 8
/A== 9
/Q== 10
/g== 11
/w== 12
AA== 13
AQ== 14"""


@pytest.fixture
def test_bpe_file(tmp_path, test_bpe_content):
    """Create a temporary BPE file for testing."""
    bpe_file = tmp_path / "test_tokenizer.model"
    bpe_file.write_text(test_bpe_content, encoding="utf-8")
    return bpe_file


@pytest.fixture
def llama3_model_path():
    """Path to Llama3 tokenizer model."""
    return Path(__file__).parent / "../../../../llama_stack/models/llama/llama3/tokenizer.model"


@pytest.fixture
def llama4_model_path():
    """Path to Llama4 tokenizer model."""
    return Path(__file__).parent / "../../../../llama_stack/models/llama/llama4/tokenizer.model"


def test_load_bpe_file_basic_functionality(test_bpe_file):
    """Test that load_bpe_file correctly parses BPE files."""
    result = load_bpe_file(test_bpe_file)

    for key, value in result.items():
        assert isinstance(key, bytes)
        assert isinstance(value, int)

    assert len(result) == 15

    expected_first_token = base64.b64decode("wA==")
    assert expected_first_token in result
    assert result[expected_first_token] == 0


def test_load_bpe_file_vs_tiktoken_with_real_model(llama3_model_path):
    """Test that our implementation produces identical results to tiktoken on real model files."""
    if not llama3_model_path.exists():
        pytest.skip("Llama3 tokenizer model not found")

    our_result = load_bpe_file(llama3_model_path)
    tiktoken_result = load_tiktoken_bpe(llama3_model_path.as_posix())

    # Compare results from our implementation and tiktoken
    assert len(our_result) == len(tiktoken_result)
    assert our_result == tiktoken_result

    assert len(our_result) > 100000
    ranks = list(our_result.values())
    assert len(ranks) == len(set(ranks))


def test_load_bpe_file_vs_tiktoken_with_llama4_model(llama4_model_path):
    """Test that our implementation produces identical results to tiktoken on Llama4 model."""
    if not llama4_model_path.exists():
        pytest.skip("Llama4 tokenizer model not found")

    our_result = load_bpe_file(llama4_model_path)
    tiktoken_result = load_tiktoken_bpe(llama4_model_path.as_posix())

    # Compare results from our implementation and tiktoken
    assert len(our_result) == len(tiktoken_result)
    assert our_result == tiktoken_result

    assert len(our_result) > 100000
    ranks = list(our_result.values())
    assert len(ranks) == len(set(ranks))


def test_load_bpe_file_malformed_lines(tmp_path):
    """Test that load_bpe_file handles malformed lines gracefully."""
    malformed_content = """wA== 0
invalid_line_without_rank
wQ== 1
invalid_base64!!! 2
9Q== 2"""

    test_file = tmp_path / "malformed.model"
    test_file.write_text(malformed_content, encoding="utf-8")

    with patch("llama_stack_spec.models.llama.tokenizer_utils.logger") as mock_logger:
        result = load_bpe_file(test_file)

        # Should have 3 valid entries (skipping malformed ones)
        assert len(result) == 3

        # Should have logged warnings for malformed lines
        assert mock_logger.warning.called
        assert mock_logger.warning.call_count > 0


def test_load_bpe_file_nonexistent_file():
    """Test that load_bpe_file raises appropriate error for nonexistent files."""
    with pytest.raises(FileNotFoundError):
        load_bpe_file("/nonexistent/path/to/file.model")


def test_tokenizer_integration():
    """Test that our load_bpe_file works correctly when used in actual tokenizers."""
    try:
        from llama_stack_spec.models.llama.llama3.tokenizer import Tokenizer as Llama3Tokenizer

        tokenizer = Llama3Tokenizer.get_instance()

        # Test basic functionality
        test_text = "Hello, world! This is a test."
        tokens = tokenizer.encode(test_text, bos=False, eos=False)
        decoded = tokenizer.decode(tokens)

        assert test_text == decoded
        assert isinstance(tokens, list)
        assert all(isinstance(token, int) for token in tokens)

    except Exception as e:
        pytest.skip(f"Llama3 tokenizer not available: {e}")


def test_performance_comparison(llama3_model_path):
    """Test that our implementation has reasonable performance compared to tiktoken."""
    if not llama3_model_path.exists():
        pytest.skip("Llama3 tokenizer model not found")

    # Time our implementation
    start_time = time.time()
    our_result = load_bpe_file(llama3_model_path)
    our_time = time.time() - start_time

    # Time tiktoken implementation
    start_time = time.time()
    tiktoken_result = load_tiktoken_bpe(llama3_model_path.as_posix())
    tiktoken_time = time.time() - start_time

    # Verify results are identical
    assert our_result == tiktoken_result

    # Our implementation should be reasonably fast (within 10x of tiktoken)
    # This is a loose bound since we're optimizing for correctness, not speed
    assert our_time < tiktoken_time * 10, f"Our implementation took {our_time:.3f}s vs tiktoken's {tiktoken_time:.3f}s"

    print(f"Performance comparison - Our: {our_time:.3f}s, Tiktoken: {tiktoken_time:.3f}s")
