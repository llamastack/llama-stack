# Mock Server Infrastructure

This directory contains mock servers for E2E telemetry testing.

## Structure

```
mocking/
├── README.md          ← You are here
├── __init__.py        ← Module exports
├── mock_base.py       ← Pydantic base class for all mocks
├── servers.py         ← Mock server implementations
└── harness.py         ← Async startup harness
```

## Files

### `mock_base.py` - Base Class
Pydantic base model that all mock servers must inherit from.

**Contract:**
```python
class MockServerBase(BaseModel):
    async def await_start(self):
        # Start server and wait until ready
        ...

    def stop(self):
        # Stop server and cleanup
        ...
```

### `servers.py` - Mock Implementations
Contains:
- **MockOTLPCollector** - Receives OTLP telemetry (port 4318)
- **MockVLLMServer** - Simulates vLLM inference API (port 8000)

### `harness.py` - Startup Orchestration
Provides:
- **MockServerConfig** - Pydantic config for server registration
- **start_mock_servers_async()** - Starts servers in parallel
- **stop_mock_servers()** - Stops all servers

## Creating a New Mock Server

### Step 1: Implement the Server

Add to `servers.py`:
```python
class MockRedisServer(MockServerBase):
    """Mock Redis server."""

    port: int = Field(default=6379)

    # Non-Pydantic fields
    server: Any = Field(default=None, exclude=True)

    def model_post_init(self, __context):
        self.server = None

    async def await_start(self):
        """Start Redis mock and wait until ready."""
        # Start your server
        self.server = create_redis_server(self.port)
        self.server.start()

        # Wait for port to be listening
        for _ in range(10):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if sock.connect_ex(("localhost", self.port)) == 0:
                sock.close()
                return  # Ready!
            await asyncio.sleep(0.1)

    def stop(self):
        if self.server:
            self.server.stop()
```

### Step 2: Register in Test

In `test_otel_e2e.py`, add to MOCK_SERVERS list:
```python
MOCK_SERVERS = [
    # ... existing servers ...
    MockServerConfig(
        name="Mock Redis",
        server_class=MockRedisServer,
        init_kwargs={"port": 6379},
    ),
]
```

### Step 3: Done!

The harness automatically:
- Creates the server instance
- Calls `await_start()` in parallel with other servers
- Returns when all are ready
- Stops all servers on teardown

## Benefits

✅ **Parallel Startup** - All servers start simultaneously
✅ **Type-Safe** - Pydantic validation
✅ **Simple** - Just implement 2 methods
✅ **Fast** - No HTTP polling, direct port checking
✅ **Clean** - Async/await pattern

## Usage in Tests

```python
@pytest.fixture(scope="module")
def mock_servers():
    servers = asyncio.run(start_mock_servers_async(MOCK_SERVERS))
    yield servers
    stop_mock_servers(servers)


# Access specific servers
@pytest.fixture(scope="module")
def mock_redis(mock_servers):
    return mock_servers["Mock Redis"]
```

## Key Design Decisions

### Why Pydantic?
- Type safety for server configuration
- Built-in validation
- Clear interface contract

### Why `await_start()` instead of HTTP `/ready`?
- Faster (no HTTP round-trip)
- Simpler (direct port checking)
- More reliable (internal state, not external endpoint)

### Why separate harness?
- Reusable across different test files
- Easy to add new servers
- Centralized error handling

## Examples

See `test_otel_e2e.py` for real-world usage:
- Line ~200: MOCK_SERVERS configuration
- Line ~230: Convenience fixtures
- Line ~240: Using servers in tests

