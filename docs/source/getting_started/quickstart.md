## Quickstart

Get started with Llama Stack in minutes!

Llama Stack is a stateful service with REST APIs to support the seamless transition of AI applications across different
environments. You can build and test using a local server first and deploy to a hosted endpoint for production.

In this guide, we'll walk through how to build a RAG application locally using Llama Stack with [Ollama](https://ollama.com/)
as the inference [provider](../providers/inference/index) for a Llama Model.

**💡 Notebook Version:** You can also follow this quickstart guide in a Jupyter notebook format: [quick_start.ipynb](https://github.com/meta-llama/llama-stack/blob/main/docs/quick_start.ipynb)

#### Step 1: Install and setup
1. Install [uv](https://docs.astral.sh/uv/)
2. Run inference on a Llama model with [Ollama](https://ollama.com/download)
```bash
ollama run llama3.2:3b --keepalive 60m
```

#### Step 2: Run the Llama Stack server

We will use `uv` to run the Llama Stack server.
```bash
OLLAMA_URL=http://localhost:11434 \
  uv run --with llama-stack llama stack build --distro starter --image-type venv --run
```
#### Step 3: Run the demo
Now open up a new terminal and copy the following script into a file named `demo_script.py`.

```{literalinclude} ./demo_script.py
:language: python
```
We will use `uv` to run the script
```
uv run --with llama-stack-client,fire,requests demo_script.py
```
And you should see output like below.
```
rag_tool> Ingesting document: https://www.paulgraham.com/greatwork.html

prompt> How do you do great work?

inference> [knowledge_search(query="What is the key to doing great work")]

tool_execution> Tool:knowledge_search Args:{'query': 'What is the key to doing great work'}

tool_execution> Tool:knowledge_search Response:[TextContentItem(text='knowledge_search tool found 5 chunks:\nBEGIN of knowledge_search tool results.\n', type='text'), TextContentItem(text="Result 1:\nDocument_id:docum\nContent:  work. Doing great work means doing something important\nso well that you expand people's ideas of what's possible. But\nthere's no threshold for importance. It's a matter of degree, and\noften hard to judge at the time anyway.\n", type='text'), TextContentItem(text="Result 2:\nDocument_id:docum\nContent:  work. Doing great work means doing something important\nso well that you expand people's ideas of what's possible. But\nthere's no threshold for importance. It's a matter of degree, and\noften hard to judge at the time anyway.\n", type='text'), TextContentItem(text="Result 3:\nDocument_id:docum\nContent:  work. Doing great work means doing something important\nso well that you expand people's ideas of what's possible. But\nthere's no threshold for importance. It's a matter of degree, and\noften hard to judge at the time anyway.\n", type='text'), TextContentItem(text="Result 4:\nDocument_id:docum\nContent:  work. Doing great work means doing something important\nso well that you expand people's ideas of what's possible. But\nthere's no threshold for importance. It's a matter of degree, and\noften hard to judge at the time anyway.\n", type='text'), TextContentItem(text="Result 5:\nDocument_id:docum\nContent:  work. Doing great work means doing something important\nso well that you expand people's ideas of what's possible. But\nthere's no threshold for importance. It's a matter of degree, and\noften hard to judge at the time anyway.\n", type='text'), TextContentItem(text='END of knowledge_search tool results.\n', type='text')]

inference> Based on the search results, it seems that doing great work means doing something important so well that you expand people's ideas of what's possible. However, there is no clear threshold for importance, and it can be difficult to judge at the time.

To further clarify, I would suggest that doing great work involves:

* Completing tasks with high quality and attention to detail
* Expanding on existing knowledge or ideas
* Making a positive impact on others through your work
* Striving for excellence and continuous improvement

Ultimately, great work is about making a meaningful contribution and leaving a lasting impression.
```
Congratulations! You've successfully built your first RAG application using Llama Stack! 🎉🥳

```{admonition} HuggingFace access
:class: tip

If you are getting a **401 Client Error** from HuggingFace for the **all-MiniLM-L6-v2** model, try setting **HF_TOKEN** to a valid HuggingFace token in your environment
```

### Next Steps

Now you're ready to dive deeper into Llama Stack!
- Explore the [Detailed Tutorial](./detailed_tutorial.md).
- Try the [Getting Started Notebook](https://github.com/meta-llama/llama-stack/blob/main/docs/getting_started.ipynb).
- Browse more [Notebooks on GitHub](https://github.com/meta-llama/llama-stack/tree/main/docs/notebooks).
- Learn about Llama Stack [Concepts](../concepts/index.md).
- Discover how to [Build Llama Stacks](../distributions/index.md).
- Refer to our [References](../references/index.md) for details on the Llama CLI and Python SDK.
- Check out the [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repository for example applications and tutorials.
