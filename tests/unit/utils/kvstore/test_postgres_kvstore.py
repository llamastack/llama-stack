# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import MagicMock, patch

import pytest

from ogx.core.storage.kvstore.postgres.postgres import PostgresKVStoreImpl


def _make_config(namespace=None, table_name="ogx_kvstore"):
    """Build a PostgresKVStoreConfig without hitting the real import of psycopg2."""
    from ogx.core.storage.datatypes import PostgresKVStoreConfig

    return PostgresKVStoreConfig(
        host="localhost",
        port=5432,
        db="testdb",
        user="testuser",
        password="testpass",
        table_name=table_name,
        namespace=namespace,
    )


def _mock_cursor():
    """Return a MagicMock that behaves like a psycopg2 DictCursor."""
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    return cursor


@pytest.fixture
def mock_pg():
    """Patch psycopg2.connect and yield (mock_psycopg2, mock_conn, cursor)."""
    with patch("ogx.core.storage.kvstore.postgres.postgres.psycopg2") as mock_psycopg2:
        mock_conn = MagicMock()
        cursor = _mock_cursor()
        mock_conn.cursor.return_value = cursor
        mock_psycopg2.connect.return_value = mock_conn

        yield mock_psycopg2, mock_conn, cursor


async def _init_store(mock_pg, namespace=None):
    """Create and initialise a PostgresKVStoreImpl with a mocked connection."""
    _, _, cursor = mock_pg
    config = _make_config(namespace=namespace)
    store = PostgresKVStoreImpl(config)
    await store.initialize()
    cursor.reset_mock()
    return store, cursor


# -- 1. keys_in_range filters expired rows --


async def test_keys_in_range_sql_should_filter_expired_rows(mock_pg):
    """keys_in_range must include the expiration guard so expired rows are excluded."""
    store, cursor = await _init_store(mock_pg)
    cursor.fetchall.return_value = []

    await store.keys_in_range("a", "z")

    sql = cursor.execute.call_args[0][0]
    assert "expiration" in sql, "keys_in_range SQL must filter on expiration"
    assert "NOW()" in sql, "keys_in_range SQL must compare expiration against NOW()"


# -- 2. keys_in_range returns non-expired rows --


async def test_keys_in_range_returns_non_expired_rows(mock_pg):
    """Rows with NULL or future expiration should be returned."""
    store, cursor = await _init_store(mock_pg)
    cursor.fetchall.return_value = [["key1"], ["key2"], ["key3"]]

    result = await store.keys_in_range("a", "z")

    assert result == ["key1", "key2", "key3"]


# -- 3. keys_in_range and values_in_range return consistent key sets --


async def test_range_queries_use_same_expiration_filter(mock_pg):
    """Both keys_in_range and values_in_range must apply the same expiration guard."""
    store, cursor = await _init_store(mock_pg)
    cursor.fetchall.return_value = []

    await store.keys_in_range("a", "z")
    keys_sql = cursor.execute.call_args[0][0]

    cursor.reset_mock()
    cursor.fetchall.return_value = []

    await store.values_in_range("a", "z")
    values_sql = cursor.execute.call_args[0][0]

    keys_has_expiration = "expiration IS NULL OR expiration > NOW()" in keys_sql
    values_has_expiration = "expiration IS NULL OR expiration > NOW()" in values_sql

    assert keys_has_expiration, "keys_in_range must include expiration filtering"
    assert values_has_expiration, "values_in_range must include expiration filtering"


# -- 4. keys_in_range uses half-open range --


async def test_keys_in_range_half_open_range(mock_pg):
    """start_key is inclusive (>=), end_key is exclusive (<)."""
    store, cursor = await _init_store(mock_pg)
    cursor.fetchall.return_value = []

    await store.keys_in_range("abc", "def")

    sql = cursor.execute.call_args[0][0]
    params = cursor.execute.call_args[0][1]

    normalized_sql = " ".join(sql.split())
    assert "key >= %s" in normalized_sql, "start_key must be inclusive (>=)"
    assert "key < %s" in normalized_sql, "end_key must be exclusive (<)"
    assert "key <= %s" not in normalized_sql, "end_key must not be inclusive (<=)"
    assert params == ("abc", "def")


# -- 5. keys_in_range results are ordered by key --


async def test_keys_in_range_ordered_by_key(mock_pg):
    """keys_in_range must ORDER BY key for deterministic results."""
    store, cursor = await _init_store(mock_pg)
    cursor.fetchall.return_value = []

    await store.keys_in_range("a", "z")

    sql = " ".join(cursor.execute.call_args[0][0].upper().split())
    assert "ORDER BY KEY" in sql, "keys_in_range must include ORDER BY key"


# -- 6. get returns None for expired keys --


async def test_get_returns_none_for_expired_keys(mock_pg):
    """get() filters expired rows via SQL; a fetchone returning None means expired."""
    store, cursor = await _init_store(mock_pg)
    cursor.fetchone.return_value = None

    result = await store.get("expired_key")

    assert result is None
    sql = cursor.execute.call_args[0][0]
    assert "expiration" in sql


async def test_get_returns_value_for_valid_key(mock_pg):
    """get() returns the value when the row is not expired."""
    store, cursor = await _init_store(mock_pg)
    cursor.fetchone.return_value = ["hello"]

    result = await store.get("valid_key")

    assert result == "hello"


# -- 7. Namespace prefixing in range queries --


async def test_namespace_prefix_applied_in_keys_in_range(mock_pg):
    """When namespace is set, keys_in_range must prefix start_key and end_key."""
    store, cursor = await _init_store(mock_pg, namespace="ns")
    cursor.fetchall.return_value = []

    await store.keys_in_range("a", "z")

    params = cursor.execute.call_args[0][1]
    assert params == ("ns:a", "ns:z")


async def test_namespace_prefix_applied_in_values_in_range(mock_pg):
    """When namespace is set, values_in_range must prefix start_key and end_key."""
    store, cursor = await _init_store(mock_pg, namespace="ns")
    cursor.fetchall.return_value = []

    await store.values_in_range("a", "z")

    params = cursor.execute.call_args[0][1]
    assert params == ("ns:a", "ns:z")


async def test_no_namespace_leaves_keys_unmodified(mock_pg):
    """When namespace is None, keys should not be prefixed."""
    store, cursor = await _init_store(mock_pg, namespace=None)
    cursor.fetchall.return_value = []

    await store.keys_in_range("start", "end")

    params = cursor.execute.call_args[0][1]
    assert params == ("start", "end")


async def test_namespace_prefix_applied_in_get(mock_pg):
    """get() must prefix the key with the namespace."""
    store, cursor = await _init_store(mock_pg, namespace="ns")
    cursor.fetchone.return_value = ["value"]

    await store.get("mykey")

    params = cursor.execute.call_args[0][1]
    assert params == ("ns:mykey",)


async def test_namespace_prefix_applied_in_set(mock_pg):
    """set() must prefix the key with the namespace."""
    store, cursor = await _init_store(mock_pg, namespace="ns")

    await store.set("mykey", "myvalue")

    params = cursor.execute.call_args[0][1]
    assert params[0] == "ns:mykey"


async def test_namespace_prefix_applied_in_delete(mock_pg):
    """delete() must prefix the key with the namespace."""
    store, cursor = await _init_store(mock_pg, namespace="ns")

    await store.delete("mykey")

    params = cursor.execute.call_args[0][1]
    assert params == ("ns:mykey",)
