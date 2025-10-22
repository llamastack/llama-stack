from llama_stack_client import LlamaStackClient
from llama_stack_client import Agent, AgentEventLogger
from rich.pretty import pprint
import uuid

client = LlamaStackClient(base_url=f"http://localhost:8321",
    default_headers={
        "X-Telemetry-Service": "llama-stack-agent",
        "X-Telemetry-Version": "1.0.0",
    })

models = client.models.list()
llm = next(m for m in models if m.model_type == "llm" and m.provider_id == "vllm")
model_id = llm.identifier

agent = Agent(client, model=model_id, instructions="You are a helpful assistant.")

s_id = agent.create_session(session_name=f"s{uuid.uuid4().hex}")

print("Non-streaming ...")
response = agent.create_turn(
    messages=[{"role": "user", "content": "Who are you?"}],
    session_id=s_id,
    stream=False,
)
print("agent>", response.output_message.content)

print("Streaming ...")
stream = agent.create_turn(
    messages=[{"role": "user", "content": "Who are you?"}], session_id=s_id, stream=True
)
for event in stream:
    pprint(event)

print("Streaming with print helper...")
stream = agent.create_turn(
    messages=[{"role": "user", "content": "Who are you?"}], session_id=s_id, stream=True
)
for event in AgentEventLogger().log(stream):
    event.print()