# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from unittest.mock import patch

import pytest

from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.providers.remote.vector_io.pgvector.pgvector import PGVectorIndex

PGVECTOR_PROVIDER = "pgvector"


@pytest.fixture(scope="session")
def loop():
    return asyncio.new_event_loop()


@pytest.fixture
def embedding_dimension():
    """Default embedding dimension for tests."""
    return 384


@pytest.fixture
async def pgvector_index(embedding_dimension, mock_psycopg2_connection):
    """Create a PGVectorIndex instance with mocked database connection."""
    connection, cursor = mock_psycopg2_connection

    vector_db = VectorDB(
        identifier="test-vector-db",
        embedding_model="test-model",
        embedding_dimension=embedding_dimension,
        provider_id=PGVECTOR_PROVIDER,
        provider_resource_id=f"{PGVECTOR_PROVIDER}:test-vector-db",
    )

    with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.psycopg2"):
        # Use explicit COSINE distance metric for consistent testing
        index = PGVectorIndex(vector_db, embedding_dimension, connection, distance_metric="COSINE")

    return index, cursor


class TestPGVectorIndex:
    def test_distance_metric_validation(self, embedding_dimension, mock_psycopg2_connection):
        connection, cursor = mock_psycopg2_connection

        vector_db = VectorDB(
            identifier="test-vector-db",
            embedding_model="test-model",
            embedding_dimension=embedding_dimension,
            provider_id=PGVECTOR_PROVIDER,
            provider_resource_id=f"{PGVECTOR_PROVIDER}:test-vector-db",
        )

        with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.psycopg2"):
            index = PGVectorIndex(vector_db, embedding_dimension, connection, distance_metric="L2")
            assert index.distance_metric == "L2"
            with pytest.raises(ValueError, match="Distance metric 'INVALID' is not supported"):
                PGVectorIndex(vector_db, embedding_dimension, connection, distance_metric="INVALID")

    def test_get_pgvector_search_function(self, pgvector_index):
        index, cursor = pgvector_index
        supported_metrics = index.PGVECTOR_DISTANCE_METRIC_TO_SEARCH_FUNCTION

        for metric, function in supported_metrics.items():
            index.distance_metric = metric
            assert index.get_pgvector_search_function() == function

    def test_check_distance_metric_availability(self, pgvector_index):
        index, cursor = pgvector_index
        supported_metrics = index.PGVECTOR_DISTANCE_METRIC_TO_SEARCH_FUNCTION

        for metric in supported_metrics:
            index.check_distance_metric_availability(metric)

        with pytest.raises(ValueError, match="Distance metric 'INVALID' is not supported"):
            index.check_distance_metric_availability("INVALID")

    def test_constructor_invalid_distance_metric(self, embedding_dimension, mock_psycopg2_connection):
        connection, cursor = mock_psycopg2_connection

        vector_db = VectorDB(
            identifier="test-vector-db",
            embedding_model="test-model",
            embedding_dimension=embedding_dimension,
            provider_id=PGVECTOR_PROVIDER,
            provider_resource_id=f"{PGVECTOR_PROVIDER}:test-vector-db",
        )

        with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.psycopg2"):
            with pytest.raises(ValueError, match="Distance metric 'INVALID_METRIC' is not supported by PGVector"):
                PGVectorIndex(vector_db, embedding_dimension, connection, distance_metric="INVALID_METRIC")

            with pytest.raises(ValueError, match="Supported metrics are:"):
                PGVectorIndex(vector_db, embedding_dimension, connection, distance_metric="UNKNOWN")

            try:
                index = PGVectorIndex(vector_db, embedding_dimension, connection, distance_metric="COSINE")
                assert index.distance_metric == "COSINE"
            except ValueError:
                pytest.fail("Valid distance metric 'COSINE' should not raise ValueError")

    def test_constructor_all_supported_distance_metrics(self, embedding_dimension, mock_psycopg2_connection):
        connection, cursor = mock_psycopg2_connection

        vector_db = VectorDB(
            identifier="test-vector-db",
            embedding_model="test-model",
            embedding_dimension=embedding_dimension,
            provider_id=PGVECTOR_PROVIDER,
            provider_resource_id=f"{PGVECTOR_PROVIDER}:test-vector-db",
        )

        supported_metrics = ["L2", "L1", "COSINE", "INNER_PRODUCT", "HAMMING", "JACCARD"]

        with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.psycopg2"):
            for metric in supported_metrics:
                try:
                    index = PGVectorIndex(vector_db, embedding_dimension, connection, distance_metric=metric)
                    assert index.distance_metric == metric

                    expected_operators = {
                        "L2": "<->",
                        "L1": "<+>",
                        "COSINE": "<=>",
                        "INNER_PRODUCT": "<#>",
                        "HAMMING": "<~>",
                        "JACCARD": "<%>",
                    }
                    assert index.get_pgvector_search_function() == expected_operators[metric]
                except Exception as e:
                    pytest.fail(f"Valid distance metric '{metric}' should not raise exception: {e}")
