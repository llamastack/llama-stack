# inline::sqlite_vec

## Description


Please refer to the sqlite-vec provider documentation.


## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `db_path` | `<class 'str'>` | No |  | Path to the SQLite database file |
| `kvstore` | `utils.kvstore.config.RedisKVStoreConfig \| utils.kvstore.config.SqliteKVStoreConfig \| utils.kvstore.config.PostgresKVStoreConfig \| utils.kvstore.config.MongoDBKVStoreConfig` | No | sqlite | Config for KV store backend (SQLite only for now) |

## Sample Configuration

```yaml
db_path: ${env.SQLITE_STORE_DIR:=~/.llama/dummy}/sqlite_vec.db
kvstore:
  type: sqlite
  db_path: ${env.SQLITE_STORE_DIR:=~/.llama/dummy}/sqlite_vec_registry.db

```

## Deprecation Notice

```{warning}
Please use the `inline::sqlite-vec` provider (notice the hyphen instead of underscore) instead.
```

