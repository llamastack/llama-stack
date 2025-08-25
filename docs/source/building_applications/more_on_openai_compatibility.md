# More on Llama Stack's OpenAI API Compatibility and other AI Frameworks

Many of the other Agentic frameworks also recognize the value of providing OpenAI API compatibility to allow for coupling
with their framework specific APIs, similar to the use of the OpenAI Responses API from a Llama Stack Client
instance as described in the previous [Agents vs Responses API](responses_vs_agents) section.

This OpenAI API compatibility becomes the "least common denominator" of sorts, and allows for migrating these agentic applications written
with these other frameworks onto AI infrastructure leveraging Llama Stack.  Once on Llama Stack, the application maintainer
can then leverage all the advantages Llama Stack can provide as summarized in the [Core Concepts section](../concepts/index.md).

As the Llama Stack community continues to dive into these different AI Frameworks with Open AI API compatibility, a
variety of documentation sections, examples, and references will be provided.  Here is what is currently available:

- **[LangChain/LangGraph](langchain_langgraph/index)**: the LangChain and associated LangGraph AI Frameworks.

```{toctree}
:hidden:
:maxdepth: 1

langchain_langgraph/index
```