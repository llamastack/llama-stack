# Agents vs OpenAI Responses API

Llama Stack (LLS) provides two different APIs for building AI applications with tool calling capabilities: the **Agents API** and the **OpenAI Responses API**. While both enable AI systems to use tools, and maintain full conversation history, they serve different use cases and have distinct characteristics.

```{note}
 **Note:** For simple and basic inferencing, you may want to use the [Chat Completions API](../providers/openai.md#chat-completions) directly, before progressing to Agents or Responses API.
```

## Overview

### LLS Agents API
The Agents API is a full-featured, stateful system designed for complex, multi-turn conversations. It maintains conversation state through persistent sessions identified by a unique session ID. The API supports comprehensive agent lifecycle management, detailed execution tracking, and rich metadata about each interaction through a structured session/turn/step hierarchy. The API can orchestrate multiple tool calls within a single turn.

### OpenAI Responses API
The OpenAI Responses API is a full-featured, stateful system designed for complex, multi-turn conversations, with direct compatibility with OpenAI's conversational patterns enhanced by LLama Stack's tool calling capabilities. It maintains conversation state by chaining responses through a `previous_response_id`, allowing interactions to branch or continue from any prior point. Each response can perform multiple tool calls within a single turn.

### Key Differences
The LLS Agents API uses the Chat Completions API on the backend for inference as it's the industry standard for building AI applications and most LLM providers are compatible with this API. For a detailed comparison between Responses and Chat Completions, see [OpenAI's documentation](https://platform.openai.com/docs/guides/responses-vs-chat-completions).

Additionally, Agents let you specify input/output shields whereas Responses do not (though support is planned). Agents use a linear conversation model referenced by a single session ID. Responses, on the other hand, support branching, where each response can serve as a fork point, and conversations are tracked by the latest response ID. Responses also lets you dynamically choose the model, vector store, files, MCP servers, and more on each inference call, enabling more complex workflows. Agents require a static configuration for these components at the start of the session.

Today the Agents and Responses APIs can be used independently depending on the use case. But, it is also productive to treat the APIs as complementary. It is not currently supported, but it is planned for the LLS Agents API to alternatively use the Responses API as its backend instead of the default Chat Completions API, i.e., enabling a combination of the safety features of Agents with the dynamic configuration and branching capabilities of Responses.

| Feature | LLS Agents API | OpenAI Responses API |
|---------|------------|---------------------|
| **Conversation Management** | Linear persistent sessions | Can branch from any previous response ID |
| **Input/Output Safety Shields** | Supported | Not yet supported |
| **Per-call Flexibility** | Static per-session configuration | Dynamic per-call configuration |

## Use Case Example: Research with Multiple Search Methods

Let's compare how both APIs handle a research task where we need to:
1. Search for current information and examples
2. Access different information sources dynamically
3. Continue the conversation based on search results

### Agents API: Session-based configuration with safety shields

```python
# Create agent with static session configuration
agent = Agent(
    client,
    model="Llama3.2-3B-Instruct",
    instructions="You are a helpful coding assistant",
    tools=[
        {
            "name": "builtin::rag/knowledge_search",
            "args": {"vector_db_ids": ["code_docs"]},
        },
        "builtin::code_interpreter",
    ],
    input_shields=["llama_guard"],
    output_shields=["llama_guard"],
)

session_id = agent.create_session("code_session")

# First turn: Search and execute
response1 = agent.create_turn(
    messages=[
        {
            "role": "user",
            "content": "Find examples of sorting algorithms and run a bubble sort on [3,1,4,1,5]",
        },
    ],
    session_id=session_id,
)

# Continue conversation in same session
response2 = agent.create_turn(
    messages=[
        {
            "role": "user",
            "content": "Now optimize that code and test it with a larger dataset",
        },
    ],
    session_id=session_id,  # Same session, maintains full context
)

# Agents API benefits:
# ✅ Safety shields protect against malicious code execution
# ✅ Session maintains context between code executions
# ✅ Consistent tool configuration throughout conversation
print(f"First result: {response1.output_message.content}")
print(f"Optimization: {response2.output_message.content}")
```

### Responses API: Dynamic per-call configuration with branching

```python
# First response: Use web search for latest algorithms
response1 = client.responses.create(
    model="Llama3.2-3B-Instruct",
    input="Search for the latest efficient sorting algorithms and their performance comparisons",
    tools=[
        {
            "type": "web_search",
        },
    ],  # Web search for current information
)

# Continue conversation: Switch to file search for local docs
response2 = client.responses.create(
    model="Llama3.2-1B-Instruct",  # Switch to faster model
    input="Now search my uploaded files for existing sorting implementations",
    tools=[
        {  # Using Responses API built-in tools
            "type": "file_search",
            "vector_store_ids": ["vs_abc123"],  # Vector store containing uploaded files
        },
    ],
    previous_response_id=response1.id,
)

# Branch from first response: Try different search approach
response3 = client.responses.create(
    model="Llama3.2-3B-Instruct",
    input="Instead, search the web for Python-specific sorting best practices",
    tools=[{"type": "web_search"}],  # Different web search query
    previous_response_id=response1.id,  # Branch from response1
)

# Responses API benefits:
# ✅ Dynamic tool switching (web search ↔ file search per call)
# ✅ OpenAI-compatible tool patterns (web_search, file_search)
# ✅ Branch conversations to explore different information sources
# ✅ Model flexibility per search type
print(f"Web search results: {response1.output_message.content}")
print(f"File search results: {response2.output_message.content}")
print(f"Alternative web search: {response3.output_message.content}")
```

Both APIs demonstrate distinct strengths that make them valuable on their own for different scenarios. The Agents API excels in providing structured, safety-conscious workflows with persistent session management, while the Responses API offers flexibility through dynamic configuration and OpenAI compatible tool patterns.

## Use Case Examples

### 1. **Research and Analysis with Safety Controls**
**Best Choice: Agents API**

**Scenario:** You're building a research assistant for a financial institution that needs to analyze market data, execute code to process financial models, and search through internal compliance documents. The system must ensure all interactions are logged for regulatory compliance and protected by safety shields to prevent malicious code execution or data leaks.

**Why Agents API?** The Agents API provides persistent session management for iterative research workflows, built-in safety shields to protect against malicious code in financial models, and structured execution logs (session/turn/step) required for regulatory compliance. The static tool configuration ensures consistent access to your knowledge base and code interpreter throughout the entire research session.

### 2. **Dynamic Information Gathering with Branching Exploration**
**Best Choice: Responses API**

**Scenario:** You're building a competitive intelligence tool that helps businesses research market trends. Users need to dynamically switch between web search for current market data and file search through uploaded industry reports. They also want to branch conversations to explore different market segments simultaneously and experiment with different models for various analysis types.

**Why Responses API?** The Responses API's branching capability lets users explore multiple market segments from any research point. Dynamic per-call configuration allows switching between web search and file search as needed, while experimenting with different models (faster models for quick searches, more powerful models for deep analysis). The OpenAI-compatible tool patterns make integration straightforward.

### 3. **OpenAI Migration with Advanced Tool Capabilities**
**Best Choice: Responses API**

**Scenario:** You have an existing application built with OpenAI's Assistants API that uses file search and web search capabilities. You want to migrate to Llama Stack for better performance and cost control while maintaining the same tool calling patterns and adding new capabilities like dynamic vector store selection.

**Why Responses API?** The Responses API provides full OpenAI tool compatibility (`web_search`, `file_search`) with identical syntax, making migration seamless. The dynamic per-call configuration enables advanced features like switching vector stores per query or changing models based on query complexity - capabilities that extend beyond basic OpenAI functionality while maintaining compatibility.

### 4. **Educational Programming Tutor**
**Best Choice: Agents API**

**Scenario:** You're building a programming tutor that maintains student context across multiple sessions, safely executes code exercises, and tracks learning progress with audit trails for educators.

**Why Agents API?** Persistent sessions remember student progress across multiple interactions, safety shields prevent malicious code execution while allowing legitimate programming exercises, and structured execution logs help educators track learning patterns.

### 5. **Advanced Software Debugging Assistant**
**Best Choice: Agents API with Responses Backend**

**Scenario:** You're building a debugging assistant that helps developers troubleshoot complex issues. It needs to maintain context throughout a debugging session, safely execute diagnostic code, switch between different analysis tools dynamically, and branch conversations to explore multiple potential causes simultaneously.

**Why Agents + Responses?** The Agent provides safety shields for code execution and session management for the overall debugging workflow. The underlying Responses API enables dynamic model selection and flexible tool configuration per query, while branching lets you explore different theories (memory leak vs. concurrency issue) from the same debugging point and compare results.

> **Note:** The ability to use Responses API as the backend for Agents is not yet implemented but is planned for a future release. Currently, Agents use Chat Completions API as their backend by default.

## For More Information

- **LLS Agents API**: For detailed information on creating and managing agents, see the [Agents documentation](agent.md)
- **OpenAI Responses API**: For information on using the OpenAI-compatible responses API, see the [OpenAI API documentation](https://platform.openai.com/docs/api-reference/responses)
- **Chat Completions API**: For the default backend API used by Agents, see the [Chat Completions providers documentation](../providers/openai.md#chat-completions)
- **Agent Execution Loop**: For understanding how agents process turns and steps in their execution, see the [Agent Execution Loop documentation](agent_execution_loop.md)
