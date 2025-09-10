# remote::s3

## Description

AWS S3-based file storage provider for scalable cloud file management with metadata persistence.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `bucket_name` | `<class 'str'>` | No |  | S3 bucket name to store files |
| `region` | `<class 'str'>` | No | us-east-1 | AWS region where the bucket is located |
| `aws_access_key_id` | `str \| None` | No |  | AWS access key ID (optional if using IAM roles) |
| `aws_secret_access_key` | `str \| None` | No |  | AWS secret access key (optional if using IAM roles) |
| `endpoint_url` | `str \| None` | No |  | Custom S3 endpoint URL (for MinIO, LocalStack, etc.) |
| `auto_create_bucket` | `<class 'bool'>` | No | False | Automatically create the S3 bucket if it doesn't exist |
| `metadata_store` | `utils.sqlstore.sqlstore.SqliteSqlStoreConfig \| utils.sqlstore.sqlstore.PostgresSqlStoreConfig` | No | sqlite | SQL store configuration for file metadata |

## Sample Configuration

```yaml
bucket_name: ${env.S3_BUCKET_NAME}
region: ${env.AWS_REGION:=us-east-1}
aws_access_key_id: ${env.AWS_ACCESS_KEY_ID:=}
aws_secret_access_key: ${env.AWS_SECRET_ACCESS_KEY:=}
endpoint_url: ${env.S3_ENDPOINT_URL:=}
auto_create_bucket: ${env.S3_AUTO_CREATE_BUCKET:=false}
metadata_store:
  type: sqlite
  db_path: ${env.SQLITE_STORE_DIR:=~/.llama/dummy}/s3_files_metadata.db

```

