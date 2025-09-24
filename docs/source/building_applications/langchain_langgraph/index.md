# OpenAI, LangChain, and LangGraph via Llama Stack

One popular AI framework that exposes Open AI API compatibility is LangChain, with its [OpenAI Provider](https://python.langchain.com/docs/integrations/providers/openai/).

With LangChain's OpenAI API compatibility, and using the Llama Stack OpenAI-compatible endpoint URL (`http://localhost:8321/v1/openapi/v1`, for example, if you are running Llama Stack
locally) as the Open AI API provider, you can run your existing LangChain AI applications in your Llama Stack environment.

There is also LangGraph, an associated by separate extension to the LangChain framework, to consider.  While LangChain is excellent for creating
linear sequences of operations (chains), LangGraph allows for more dynamic workflows (graphs) with loops, branching, and persistent state.
This makes LangGraph ideal for sophisticated agent-based systems where the flow of control is not predetermined.
You can use your existing LangChain components in combination with LangGraph components to create more complex,
multi-agent applications.

As this LangChain/LangGraph section of the Llama Stack docs iterates and expands, a variety of samples that vary both in

- How complex the application is
- What aspects of Llama Stack are leveraged in conjunction with the application

will be provided, as well as references to third party sites with samples.

Local examples:

- **[Starter](langchain_langgraph)**:  Explore a simple, graph-based agentic application that exposes a simple tool to add numbers together.

External sites:

- **[Responses](more_on_responses)**:  A deeper dive into the newer OpenAI Responses API (vs. the Chat Completion API).


```{toctree}
:hidden:
:maxdepth: 1

langchain_langgraph
more_on_responses
```