from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321",
    default_headers={
        "X-Telemetry-Service": "llama-stack-inference",
        "X-Telemetry-Version": "1.0.0",
    }
    )

# List available models
models = client.models.list()

# Select the first LLM
llm = next(m for m in models if m.model_type == "llm" and m.provider_id == "vllm")
model_id = llm.identifier

print("Model:", model_id)

response = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding"},
    ],
)
print(response)