# Job Scheduler

## Run Config → Adapter Init Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Run Config YAML                                       │
├─────────────────────────────────────────────────────────┤
│ providers:                                               │
│   job_scheduler:                                         │
│     - provider_type: inline::scheduler                   │
│       config: { kvstore: {...} }                         │
│   vector_io:                                             │
│     - provider_type: inline::faiss                       │
│       config: { persistence: {...} }                     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Stack.initialize()                                    │
│    → resolve_impls(run_config, ...)                     │
│    → instantiate_providers(sorted_by_deps)              │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 3. For each provider (in dependency order):              │
│                                                           │
│    A. Job Scheduler (no dependencies)                    │
│       ├─ get_provider_impl(config, deps={})             │
│       ├─ scheduler = InlineSchedulerImpl(config)        │
│       └─ await scheduler.initialize()                   │
│                                                           │
│    B. Vector IO (depends on inference, job_scheduler)    │
│       ├─ deps = {                                        │
│       │     Api.inference: <impl>,                       │
│       │     Api.job_scheduler: <scheduler_impl>          │
│       │   }                                               │
│       ├─ get_provider_impl(config, deps)                │
│       │   ├─ adapter = FaissVectorIOAdapter(            │
│       │   │     config,                                  │
│       │   │     deps[Api.inference],                     │
│       │   │     deps.get(Api.files),                     │
│       │   │     deps.get(Api.job_scheduler)  ← Here!    │
│       │   │   )                                           │
│       │   └─ await adapter.initialize()                 │
│       └─ return adapter                                  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Adapter.__init__(scheduler)                          │
│    ├─ self.scheduler = scheduler                        │
│    └─ scheduler.register_job_executor(...)              │
└─────────────────────────────────────────────────────────┘
```

**Key Points:**
- **Dependency resolution**: Providers instantiated in topological order
- **Automatic injection**: Framework passes `deps[Api.job_scheduler]`
- **Registry declares it**: `optional_api_dependencies=[Api.job_scheduler]`
- **User configures it**: Run YAML specifies inline vs celery

---

## Server Startup Flow

## Two-Phase Initialization

Separate scheduler initialization into two phases:
- **Phase 1 (`initialize`)**: Load jobs from storage, but don't resume them
- **Phase 2 (`start`)**: Resume jobs after all executors are registered

```
┌─────────────────────────────────────────────────────────┐
│ Stack.initialize()                                       │
├─────────────────────────────────────────────────────────┤
│   → resolve_impls()                                      │
│      ├─ Job Scheduler                                    │
│      │   └─ scheduler.initialize()  ← Phase 1           │
│      │       • Connect to KVStore                        │
│      │       • Load jobs into memory                     │
│      │       • DON'T resume jobs yet ❌                  │
│      │                                                    │
│      ├─ Vector IO Provider                              │
│      │   ├─ __init__():                                  │
│      │   │   └─ scheduler.register_job_executor(...)    │
│      │   └─ provider.initialize()                        │
│      │                                                    │
│      └─ [All providers initialized, executors ✓]        │
│                                                           │
│   → Start schedulers                                     │
│      └─ scheduler.start()  ← Phase 2                    │
│          └─ Resume incomplete jobs                       │
│              • Executors are registered ✓                │
│              • Jobs execute successfully ✓               │
└─────────────────────────────────────────────────────────┘
```

## Implementation

### 1. Scheduler Protocol (`api.py`)

```python
async def initialize(self) -> None:
    """
    Initialize the scheduler (connect to backend, load persisted jobs).

    Note: Does NOT start processing jobs. Call start() after all
    executors are registered.
    """
    ...


async def start(self) -> None:
    """
    Start processing jobs after all executors are registered.

    This resumes any incomplete jobs from storage. Must be called
    after initialize() and after all job executors have been
    registered via register_job_executor().
    """
    ...
```

### 2. Scheduler Implementation

```python
class InlineSchedulerImpl(Scheduler):
    async def initialize(self) -> None:
        self._kvstore = await kvstore_impl(self.config.kvstore)
        await self._load_jobs_from_storage()
        # DON'T call _resume_incomplete_jobs() here!

    async def start(self) -> None:
        """Call AFTER all executors are registered."""
        await self._resume_incomplete_jobs()
```

### 3. Stack Integration (`stack.py`)

```python
async def initialize(self):
    # ... existing code ...
    impls = await resolve_impls(...)  # All providers initialized

    # NEW: Start schedulers after all executors registered
    if Api.job_scheduler in impls:
        await impls[Api.job_scheduler].start()

    self.impls = impls
```

### 4. Provider Integration

```python
class VectorIOAdapter:
    def __init__(self, config, inference_api, files_api, scheduler):
        # Register executor during __init__ (before scheduler.start())
        scheduler.register_job_executor(
            "vector_store_file_batch", self._process_file_batch
        )
```

## Behavior

### Case 1: Clean Start (No Jobs)
```python
await scheduler.start()
# _resume_incomplete_jobs() finds no jobs
# Loop completes, nothing happens ✓
```

### Case 2: After Crash (Jobs Exist)
```python
await scheduler.start()
# _resume_incomplete_jobs() finds PENDING/RUNNING jobs
# Executors are registered ✓
# Creates asyncio tasks successfully ✓
# Jobs run in background ✓
```


## Usage: OpenAI Vector Store Mixin

### Current Pattern (Direct asyncio tasks)
```python
class OpenAIVectorStoreMixin:
    def __init__(self, files_api, kvstore):
        self._file_batch_tasks: dict[str, asyncio.Task] = {}

    async def openai_create_vector_store_file_batch(self, params):
        # Create background task directly
        task = asyncio.create_task(self._process_file_batch_async(params))
        self._file_batch_tasks[batch_id] = task
```

### New Pattern (Job Scheduler)
```python
class OpenAIVectorStoreMixin:
    def __init__(self, files_api, kvstore, scheduler):
        self.scheduler = scheduler
        # Register executor in __init__ (before scheduler.start())
        self.scheduler.register_job_executor(
            "vector_store_file_batch", self._execute_file_batch
        )

    async def openai_create_vector_store_file_batch(self, params):
        # Schedule job through scheduler
        job_id = await self.scheduler.schedule_job(
            job_type="vector_store_file_batch",
            job_data={
                "batch_id": batch_id,
                "vector_store_id": vector_store_id,
                "file_ids": params.file_ids,
                "attributes": params.attributes,
                "chunking_strategy": chunking_strategy.model_dump(),
            },
            metadata={"batch_id": batch_id},
        )
        return batch_object

    async def _execute_file_batch(self, job_data: dict) -> dict:
        """Executor called by scheduler."""
        batch_id = job_data["batch_id"]
        vector_store_id = job_data["vector_store_id"]
        file_ids = job_data["file_ids"]

        # Process files (existing logic from _process_file_batch_async)
        # await self._process_files_with_concurrency(...)

        return {"status": "completed", "files_processed": len(file_ids)}
```

### Benefits
- ✅ **Crash recovery**: Jobs survive server restarts
- ✅ **Persistence**: Job state stored in KVStore
- ✅ **Monitoring**: Query job status via `get_job_info(job_id)`
- ✅ **Cancellation**: Cancel jobs via `cancel_job(job_id)`
- ✅ **Clean separation**: Job scheduling decoupled from execution

## Single Worker vs Multi Worker

  ✅ Single Worker (Inline Scheduler - Not Implemented Yet)
```
  providers:
    job_scheduler:
      - provider_type: inline::scheduler
        config:
          kvstore: { ... }
          max_concurrent_jobs: 10

  Works because:
  - Jobs run in the same process via asyncio.create_task()
  - In-memory _jobs dict is shared within the process
  - Crash recovery works (jobs persist to KVStore)
```

  ---
  ✅ Multi Worker (Celery Scheduler - Not Implemented Yet)
```
  providers:
    job_scheduler:
      - provider_type: celery::scheduler
        config:
          broker_url: redis://localhost:6379/0
          result_backend: redis://localhost:6379/1
```
  Works because:
  - Shared message broker (Redis/RabbitMQ)
  - Celery handles distributed task queue
  - Workers coordinate via broker
  - Any worker can execute any job
