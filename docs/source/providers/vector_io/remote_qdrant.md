# remote::qdrant

## Description


Please refer to the inline provider documentation.


## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `location` | `str \| None` | No |  |  |
| `url` | `str \| None` | No |  |  |
| `port` | `int \| None` | No | 6333 |  |
| `grpc_port` | `<class 'int'>` | No | 6334 |  |
| `prefer_grpc` | `<class 'bool'>` | No | False |  |
| `https` | `bool \| None` | No |  |  |
| `api_key` | `str \| None` | No |  |  |
| `prefix` | `str \| None` | No |  |  |
| `timeout` | `int \| None` | No |  |  |
| `host` | `str \| None` | No |  |  |
| `kvstore` | `utils.kvstore.config.RedisKVStoreConfig \| utils.kvstore.config.SqliteKVStoreConfig \| utils.kvstore.config.PostgresKVStoreConfig \| utils.kvstore.config.MongoDBKVStoreConfig` | No | sqlite |  |

## Sample Configuration

```yaml
api_key: ${env.QDRANT_API_KEY:=}
kvstore:
  type: sqlite
  db_path: ${env.SQLITE_STORE_DIR:=~/.llama/dummy}/qdrant_registry.db

```

