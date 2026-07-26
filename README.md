While working it was a hectic job to scroll through datasheets and find the appropriate answer.So i created this:

Local, free, citation bacekd RAG assistant for microcontroller and sensor datasheets. The answer are grounded in the pdf texts no outside knowledge used.

## tech stack used:
pdf parsing via pypdf
embeddings achieved via sentence transformer (all-miniLM-L6-V2), local and free
vector stored via chromadb
llm used ollama
UI via Gradio

## Setup

install ollama and pull a model : ollama pull llama.3.2
create a virtual environment
create a requirment.txt file and install using pip
Drop your datasheets into document folder
python ingest.py (builds vector index)
python app.py (for UI)

## Why I chose RAG

RAG retrives the exact relevent text first and grounds the LLM's answer in it, which is cheaper more accurate for reference documents and updateable.