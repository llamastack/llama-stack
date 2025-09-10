# inline::localfs

## Description

Local filesystem-based file storage provider for managing files and documents locally.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `storage_dir` | `<class 'str'>` | No |  | Directory to store uploaded files |
| `metadata_store` | `utils.sqlstore.sqlstore.SqliteSqlStoreConfig \| utils.sqlstore.sqlstore.PostgresSqlStoreConfig` | No | sqlite | SQL store configuration for file metadata |
| `ttl_secs` | `<class 'int'>` | No | 31536000 |  |

## Sample Configuration

```yaml
storage_dir: ${env.FILES_STORAGE_DIR:=~/.llama/dummy/files}
metadata_store:
  type: sqlite
  db_path: ${env.SQLITE_STORE_DIR:=~/.llama/dummy}/files_metadata.db

```

