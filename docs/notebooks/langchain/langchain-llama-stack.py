import html
import os
import re
import tempfile
from typing import Any, List, Optional

import requests
from bs4 import BeautifulSoup
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from llama_stack_client import LlamaStackClient
from markdownify import markdownify
from readability import Document as ReadabilityDocument
from rich.pretty import pprint

# Global variables
client = None
llm = None
summary_chain = None
facts_chain = None
qa_chain = None
processed_docs = {}

# Prompt Templates (defined globally)
summary_template = PromptTemplate(
    input_variables=["document"],
    template="""Create a concise summary of this document in 5-10 sentences:

{document}

SUMMARY:""",
)

facts_template = PromptTemplate(
    input_variables=["document"],
    template="""Extract the most important facts from this document. List them as bullet points:

{document}

KEY FACTS:
-""",
)

qa_template = PromptTemplate(
    input_variables=["document", "question"],
    template="""Based on the following document, answer the question. If the answer isn't in the document, say so.

DOCUMENT:
{document}

QUESTION: {question}

ANSWER:""",
)


def load_document(source: str) -> str:
    is_url = source.startswith(("http://", "https://"))
    is_pdf = source.lower().endswith(".pdf")
    if is_pdf:
        return load_pdf(source, is_url=is_url)
    elif is_url:
        return load_from_url(source)
    else:
        raise ValueError(f"Unsupported format. Use URLs or PDF files.")


def load_pdf(source: str, is_url: bool = False) -> str:
    if is_url:
        response = requests.get(source)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            file_path = temp_file.name
    else:
        file_path = source
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return "\\n\\n".join([doc.page_content for doc in docs])
    finally:
        if is_url:
            os.remove(file_path)


def load_from_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DocumentLoader/1.0)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    doc = ReadabilityDocument(response.text)
    html_main = doc.summary(html_partial=True)
    soup = BeautifulSoup(html_main, "html.parser")
    for tag in soup(
        ["script", "style", "noscript", "header", "footer", "nav", "aside"]
    ):
        tag.decompose()
    md_text = markdownify(str(soup), heading_style="ATX")
    md_text = html.unescape(md_text)
    md_text = re.sub(r"\n{3,}", "\n\n", md_text).strip()
    return md_text


def process_document(source: str):
    global summary_chain, facts_chain, processed_docs

    print(f"📄 Loading document from: {source}")
    document = load_document(source)
    print(f"✅ Loaded {len(document):,} characters")
    print("\n📝 Generating summary...")
    summary = summary_chain.invoke({"document": document})["text"]
    print("Summary generated")
    print("🔍 Extracting key facts...")
    facts = facts_chain.invoke({"document": document})["text"]
    processed_docs[source] = {"document": document, "summary": summary, "facts": facts}
    print(f"\n✅ Processing complete!")
    print(f"📊 Document: {len(document):,} chars")
    print(f"📝 Summary: {summary[:100]}...")
    print(f"🔍 Facts: {facts[:1000]}...")
    return processed_docs[source]


def ask_question(question: str, source: str = None):
    """Answer questions about processed documents"""
    global qa_chain, processed_docs

    if not processed_docs:
        return "No documents processed yet. Use process_document() first."
    if source and source in processed_docs:
        doc_data = processed_docs[source]
    else:
        # Use the most recent document
        doc_data = list(processed_docs.values())[-1]
    answer = qa_chain.invoke({"document": doc_data["document"], "question": question})[
        "text"
    ]
    return answer


def interactive_demo():
    print("\n🎯 Interactive Document Processing Demo")
    print("Commands:")
    print("  load <url_or_path>  - Process a document")
    print("  ask <question>      - Ask about the document")
    print("  summary            - Show document summary")
    print("  facts              - Show extracted facts")
    print("  help               - Show commands")
    print("  quit               - Exit demo")

    while True:
        try:
            command = input("\n> ").strip()
            if command.lower() in ["quit", "exit"]:
                print("👋 Thanks for exploring LangChain chains!")
                break
            elif command.lower() == "help":
                print("\nCommands:")
                print("  load <url_or_path>  - Process a document")
                print("  ask <question>      - Ask about the document")
                print("  summary            - Show document summary")
                print("  facts              - Show extracted facts")
            elif command.startswith("load "):
                source = command[5:].strip()
                if source:
                    try:
                        process_document(source)
                    except Exception as e:
                        print(f"❌ Error processing document: {e}")
                else:
                    print("❓ Please provide a URL or file path")
            elif command.startswith("ask "):
                question = command[4:].strip()
                if question:
                    try:
                        answer = ask_question(question)
                        print(f"\n💬 Q: {question}")
                        print(f"📝 A: {answer}")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                else:
                    print("❓ Please provide a question")
            elif command.lower() == "summary":
                if processed_docs:
                    latest_doc = list(processed_docs.values())[-1]
                    print(f"\n📝 Summary:\n{latest_doc['summary']}")
                else:
                    print("❓ No documents processed yet")
            elif command.lower() == "facts":
                if processed_docs:
                    latest_doc = list(processed_docs.values())[-1]
                    print(f"\n🔍 Key Facts:\n{latest_doc['facts']}")
                else:
                    print("❓ No documents processed yet")
            else:
                print("❓ Unknown command. Type 'help' for options")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break


def main():
    global client, llm, summary_chain, facts_chain, qa_chain, processed_docs

    print("🚀 Starting LangChain + Llama Stack Document Processing Demo")

    client = LlamaStackClient(
        base_url="http://localhost:8321/",
    )
    os.environ["OPENAI_API_KEY"] = "dummy"
    os.environ["OPENAI_BASE_URL"] = "http://0.0.0.0:8321/v1/openai/v1"
    llm = ChatOpenAI(model="ollama/llama3:70b-instruct", base_url="http://localhost:8321/v1/openai/v1")

    # Test the wrapper
    test_response = llm.invoke("Can you help me with the document processing?")
    print(f"✅ LangChain wrapper working!")
    print(f"Response: {test_response.content[:100]}...")

    print("Available models:")
    for m in client.models.list():
        print(f"- {m.identifier}")

    print("----")
    print("Available shields (safety models):")
    for s in client.shields.list():
        print(s.identifier)
    print("----")

    model_id = "ollama/llama3:70b-instruct"
    # Create chains by combining our LLM with prompt templates
    summary_chain = LLMChain(llm=llm, prompt=summary_template)
    facts_chain = LLMChain(llm=llm, prompt=facts_template)
    qa_chain = LLMChain(llm=llm, prompt=qa_template)

    # Initialize storage for processed documents
    processed_docs = {}

    print("✅ Created 3 prompt templates:")
    print("  • Summary: Condenses documents into key points")
    print("  • Facts: Extracts important information as bullets")
    print("  • Q&A: Answers questions based on document content")

    # Test template formatting
    test_prompt = summary_template.format(
        document="This is a sample document about AI..."
    )
    print(f"\n📝 Example prompt: {len(test_prompt)} characters")

    # Start the interactive demo
    interactive_demo()


if __name__ == "__main__":
    main()
