# LangChain + Llama Stack Document Processing

1. **`langchain-llama-stack.py`** - Interactive CLI version
---

## 📋 Prerequisites

### System Requirements
- Python 3.12+
- Llama Stack server running on `http://localhost:8321/`
- Ollama or compatible model server

### Environment Setup
```bash
# Create and activate virtual environment
python3.12 -m venv llama-env-py312
source llama-env-py312/bin/activate

# Install dependencies
pip install llama-stack-client langchain langchain-core langchain-community beautifulsoup4 markdownify readability-lxml requests langchain_openai
```

---

## 🚀 Quick Start

### Start Llama Stack Server
Before running either version, ensure your Llama Stack server is running:
```bash
# Start Llama Stack server (example)
llama stack run your-config --port 8321
```

---

## 📖 Option 1: Interactive CLI Version (`langchain-llama-stack.py`)

### Features
- ✅ Interactive command-line interface
- ✅ Document loading from URLs and PDFs
- ✅ AI-powered summarization and fact extraction
- ✅ Question-answering based on document content
- ✅ Session-based document storage

### How to Run
```bash

# Run the interactive CLI
cd /docs/notebooks/langchain/
python langchain-llama-stack.py
```

### Usage Commands
Once running, you can use these interactive commands:

```
🎯 Interactive Document Processing Demo
Commands:
  load <url_or_path>  - Process a document
  ask <question>      - Ask about the document
  summary            - Show document summary
  facts              - Show extracted facts
  help               - Show commands
  quit               - Exit demo
```

### Example Session
```
> load https://en.wikipedia.org/wiki/Artificial_intelligence
📄 Loading document from: https://en.wikipedia.org/wiki/Artificial_intelligence
✅ Loaded 45,832 characters
📝 Generating summary...
🔍 Extracting key facts...
✅ Processing complete!

> summary
📝 Summary:
Artificial intelligence (AI) is the simulation of human intelligence...

> ask What are the main types of AI?
💬 Q: What are the main types of AI?
📝 A: Based on the document, the main types of AI include...

> facts
🔍 Key Facts:
- AI was founded as an academic discipline in 1956
- Machine learning is a subset of AI...

> quit
👋 Thanks for exploring LangChain chains!
```


#### Using curl:
```bash
# Check service status
curl http://localhost:8000/

# Process a document
curl -X POST http://localhost:8000/process \
     -H 'Content-Type: application/json' \
     -d '{"source": "https://en.wikipedia.org/wiki/Machine_learning"}'

# Ask a question
curl -X POST http://localhost:8000/ask \
     -H 'Content-Type: application/json' \
     -d '{"question": "What is machine learning?"}'

# Get summary
curl http://localhost:8000/summary

# Get facts
curl http://localhost:8000/facts

# List all processed documents
curl http://localhost:8000/docs
```

#### Using Python requests:
```python
import requests

# Process a document
response = requests.post(
    "http://localhost:8000/process",
    json={"source": "https://en.wikipedia.org/wiki/Deep_learning"}
)
print(response.json())

# Ask a question
response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What are neural networks?"}
)
print(response.json())

# Get facts
response = requests.get("http://localhost:8000/facts")
print(response.json())
```

---

## 🔧 Configuration

### Model Configuration
Both versions use these models by default:
- **Model ID**: `llama3.2:3b`
- **Llama Stack URL**: `http://localhost:8321/`

To change the model, edit the `model_id` parameter in the respective files.

### Supported Document Types
- ✅ **URLs**: Any web page (extracted using readability)
- ✅ **PDF files**: Local or remote PDF documents
- ❌ Plain text files (can be added if needed)

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Connection Refused to Llama Stack
**Error**: `Connection refused to http://localhost:8321/`
**Solution**:
- Ensure Llama Stack server is running
- Check if port 8321 is correct
- Verify network connectivity

#### 2. Model Not Found
**Error**: `Model not found: llama3.2:3b`
**Solution**:
- Check available models: `curl http://localhost:8321/models/list`
- Update `model_id` in the code to match available models


#### 4. Missing Dependencies
### Debug Mode
To enable verbose logging, add this to the beginning of either file:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📊 Performance Notes

### CLI Version
- **Pros**: Simple to use, interactive, good for testing
- **Cons**: Single-threaded, session-based only
- **Best for**: Development, testing, manual document analysis
---

## 🛑 Stopping Services

### CLI Version
- Press `Ctrl+C` or type `quit` in the interactive prompt
---

## 📝 Examples

### CLI Workflow
1. Start: `python langchain-llama-stack.py`
2. Load document: `load https://arxiv.org/pdf/2103.00020.pdf`
3. Get summary: `summary`
4. Ask questions: `ask What are the main contributions?`
5. Exit: `quit`

---

## 🤝 Contributing

To extend functionality:
1. Add new prompt templates for different analysis types
2. Support additional document formats
3. Add caching for processed documents
4. Implement user authentication for API version

---

## 📜 License

This project is for educational and research purposes.
