# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for PostgresKVStoreImpl.

Since unit tests cannot depend on a running Postgres server, these tests
use mocked psycopg2 to verify SQL query correctness, namespace prefixing,
expiration filtering, and error handling.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from ogx.core.storage.datatypes import PostgresKVStoreConfig


def _make_config(namespace: str | None = None, table_name: str = "test_kvstore") -> PostgresKVStoreConfig:
    return PostgresKVStoreConfig(
        host="localhost",
        port=5432,
        db="testdb",
        user="testuser",
        password="testpass",
        table_name=table_name,
        namespace=namespace,
    )


def _make_store_with_mock_cursor(config: PostgresKVStoreConfig):
    """Create a PostgresKVStoreImpl with a mocked cursor, bypassing initialize()."""
    from ogx.core.storage.kvstore.postgres.postgres import PostgresKVStoreImpl

    store = PostgresKVStoreImpl(config)
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    store._cursor = mock_cursor
    store._conn = mock_conn
    return store, mock_cursor, mock_conn


# -- Namespace prefixing -------------------------------------------------------


async def test_set_applies_namespace():
    store, cursor, _ = _make_store_with_mock_cursor(_make_config(namespace="quota"))
    await store.set("user:123", "5", expiration=None)

    cursor.execute.assert_called_once()
    args = cursor.execute.call_args
    assert args[0][1][0] == "quota:user:123"


async def test_get_applies_namespace():
    store, cursor, _ = _make_store_with_mock_cursor(_make_config(namespace="quota"))
    cursor.fetchone.return_value = None

    await store.get("user:123")

    args = cursor.execute.call_args
    assert args[0][1][0] == "quota:user:123"


async def test_delete_applies_namespace():
    store, cursor, _ = _make_store_with_mock_cursor(_make_config(namespace="myns"))

    await store.delete("k1")

    args = cursor.execute.call_args
    assert args[0][1][0] == "myns:k1"


async def test_no_namespace_passes_key_through():
    store, cursor, _ = _make_store_with_mock_cursor(_make_config(namespace=None))
    cursor.fetchone.return_value = None

    await store.get("raw_key")

    args = cursor.execute.call_args
    assert args[0][1][0] == "raw_key"


# -- SQL query correctness ----------------------------------------------------


async def test_get_filters_expired_keys():
    """get() SQL includes expiration > NOW() filter."""
    store, cursor, _ = _make_store_with_mock_cursor(_make_config())
    cursor.fetchone.return_value = None

    await store.get("k1")

    sql = cursor.execute.call_args[0][0]
    assert "expiration IS NULL OR expiration > NOW()" in sql


async def test_values_in_range_uses_half_open_interval():
    """values_in_range SQL uses >= start AND < end."""
    store, cursor, _ = _make_store_with_mock_cursor(_make_config())
    cursor.fetchall.return_value = []

    await store.values_in_range("a", "c")

    sql = cursor.execute.call_args[0][0]
    assert "key >= %s AND key < %s" in sql
    assert "key <= %s" not in sql


async def test_values_in_range_filters_expired():
    store, cursor, _ = _make_store_with_mock_cursor(_make_config())
    cursor.fetchall.return_value = []

    await store.values_in_range("a", "z")

    sql = cursor.execute.call_args[0][0]
    assert "expiration IS NULL OR expiration > NOW()" in sql


async def test_keys_in_range_uses_half_open_interval():
    """keys_in_range SQL uses >= start AND < end."""
    store, cursor, _ = _make_store_with_mock_cursor(_make_config())
    cursor.fetchall.return_value = []

    await store.keys_in_range("a", "c")

    sql = cursor.execute.call_args[0][0]
    assert "key >= %s AND key < %s" in sql


async def test_keys_in_range_filters_expired():
    """keys_in_range must also filter expired keys (bug fix verification)."""
    store, cursor, _ = _make_store_with_mock_cursor(_make_config())
    cursor.fetchall.return_value = []

    await store.keys_in_range("a", "z")

    sql = cursor.execute.call_args[0][0]
    assert "expiration IS NULL OR expiration > NOW()" in sql


async def test_range_queries_apply_namespace():
    store, cursor, _ = _make_store_with_mock_cursor(_make_config(namespace="ns"))
    cursor.fetchall.return_value = []

    await store.values_in_range("a", "z")

    params = cursor.execute.call_args[0][1]
    assert params[0] == "ns:a"
    assert params[1] == "ns:z"


async def test_keys_in_range_strips_namespace_from_results():
    """keys_in_range returns un-namespaced keys so callers can pass them to get()."""
    store, cursor, _ = _make_store_with_mock_cursor(_make_config(namespace="ns"))
    cursor.fetchall.return_value = [("ns:key1",), ("ns:key2",)]

    keys = await store.keys_in_range("a", "z")

    assert keys == ["key1", "key2"]


# -- Error handling ------------------------------------------------------------


async def test_cursor_or_raise_when_uninitialized():
    from ogx.core.storage.kvstore.postgres.postgres import PostgresKVStoreImpl

    config = _make_config()
    store = PostgresKVStoreImpl(config)

    with pytest.raises(RuntimeError, match="not initialized"):
        await store.get("k1")


async def test_initialize_wraps_connection_error():
    from ogx.core.storage.kvstore.postgres.postgres import PostgresKVStoreImpl

    config = _make_config()
    store = PostgresKVStoreImpl(config)

    with patch("ogx.core.storage.kvstore.postgres.postgres.psycopg2") as mock_pg:
        mock_pg.connect.side_effect = Exception("connection refused")
        with pytest.raises(RuntimeError, match="Could not connect"):
            await store.initialize()


# -- Shutdown ------------------------------------------------------------------


async def test_shutdown_closes_cursor_and_connection():
    store, cursor, conn = _make_store_with_mock_cursor(_make_config())

    await store.shutdown()

    cursor.close.assert_called_once()
    conn.close.assert_called_once()
    assert store._cursor is None
    assert store._conn is None


async def test_shutdown_idempotent():
    store, _, _ = _make_store_with_mock_cursor(_make_config())

    await store.shutdown()
    await store.shutdown()


# -- Config validation ---------------------------------------------------------
# NOTE: PostgresKVStoreConfig.validate_table_name is currently broken because
# @classmethod is stacked before @field_validator, making Pydantic ignore it.
# These tests document the DESIRED behavior; they are marked xfail until the
# validator stacking is fixed (swap to @field_validator / @classmethod order).


@pytest.mark.xfail(reason="table_name validator broken: @classmethod before @field_validator")
def test_table_name_rejects_sql_injection():
    with pytest.raises(ValueError, match="Invalid table name"):
        _make_config(table_name="users; DROP TABLE")


@pytest.mark.xfail(reason="table_name validator broken: @classmethod before @field_validator")
def test_table_name_rejects_empty():
    with pytest.raises(ValueError, match="Invalid table name"):
        _make_config(table_name="")


def test_table_name_accepts_valid():
    config = _make_config(table_name="ogx_kvstore_v2")
    assert config.table_name == "ogx_kvstore_v2"


@pytest.mark.xfail(reason="table_name validator broken: @classmethod before @field_validator")
def test_table_name_rejects_too_long():
    with pytest.raises(ValueError, match="less than 63"):
        _make_config(table_name="a" * 64)
