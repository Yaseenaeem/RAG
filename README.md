# Retrieval-Augmented Generation (RAG) Assistant

A local Retrieval-Augmented Generation (RAG) system that allows users to query information from multiple PDF documents using a hybrid retrieval pipeline and a locally running LLM.

The project is currently at **V1 — Backend RAG System**.


## Overview

The goal of this project was to understand and build the complete RAG pipeline rather than simply connecting a document to an LLM.
The system processes PDF documents, converts their content into searchable representations, retrieves relevant information using both semantic and keyword-based search, reranks the retrieved results, and provides the relevant context to a local LLM for answer generation.
The assistant also supports conversational follow-up questions through query rewriting and provides document/page sources with each answer.


## Architecture

PDF Documents
↓
Document Loading
↓
Chunking
↓
Embeddings
↓
FAISS Semantic Search
+
BM25 Keyword Search
↓
Hybrid Candidate Retrieval
↓
CrossEncoder Reranking
↓
Top Relevant Chunks
↓
Gemma 3 via Ollama
↓
Answer + Sources


## Key Features

- PDF document processing
- Recursive text chunking
- Sentence Transformer embeddings
- FAISS semantic retrieval
- BM25 keyword retrieval
- Hybrid retrieval
- CrossEncoder reranking
- Conversational query rewriting
- Local LLM inference
- Conversation memory
- Source and page attribution
- Persistent FAISS index and document chunks
- Multiple document support


## Retrieval Pipeline

### 1. Document Processing

PDF documents are loaded and split into smaller chunks using a recursive character text splitter.

### 2. Embeddings

Each chunk is converted into a numerical embedding using:

`all-MiniLM-L6-v2`

### 3. Semantic Retrieval

FAISS performs similarity search between the user's rewritten query and document embeddings.

### 4. Keyword Retrieval

BM25 provides an additional keyword-based retrieval layer to improve results for queries containing specific terms.

### 5. Hybrid Retrieval

Results from FAISS and BM25 are combined to create a candidate pool.

### 6. Reranking

A CrossEncoder reranks the candidate chunks based on query-document relevance.

### 7. Query Rewriting

For conversational follow-up questions, the local LLM rewrites the user's question into a standalone search query.

Example:

```text
Previous question:
What is the remote work policy?

Follow-up:
What about approval?

Rewritten query:
remote work policy approval
```


## Answer Generation

The top-ranked chunks are provided as context to Gemma 3 running locally through Ollama.
The model is instructed to answer only from the retrieved context.


## Sources

The system returns the document name and page number associated with the retrieved context.

Technologies
Python
FAISS
BM25
Sentence Transformers
CrossEncoder
LangChain
Ollama
Gemma 3
NumPy
PyPDF


## Documents: 
The RAG assistant was tested using publicly available sample HR policy templates, including employee handbooks, remote-work policies, and HR manuals. These documents were obtained from publicly accessible sources and are used solely as demonstration/test data for the project.


## Why Hybrid Retrieval?

Semantic search is useful for understanding the meaning of a query, while keyword search can be useful when exact terms matter.

This project combines:
FAISS
→ semantic similarity

BM25
→ keyword matching

CrossEncoder
→ relevance-based reranking

The goal is to improve the quality of the final context supplied to the LLM.

Example
User Query
What about approval?
Query Rewriting
remote work policy approval
Retrieved Document
Remote-Work-Policy.pdf
Generated Answer

The system provides an answer based on the retrieved document context and includes the relevant document/page sources.


## Limitations

This is the first version (V1) of the project and is intended as a functional foundation for a broader RAG system. While the system can retrieve relevant information and avoid generating answers when sufficient evidence is not found, retrieval may occasionally include semantically related but irrelevant chunks, and the displayed sources represent retrieved documents rather than guaranteed direct evidence for every answer. Future versions can improve document parsing and chunking, metadata-aware retrieval, reranking and context filtering, and more precise source attribution.



## Project Status
V1 — Backend RAG ✓

Next — V2

Planned:
User interface
Improved user experience
Cleaner source presentation
Application-level interaction


## What I Learned: 
This project was built to understand the reasoning behind a RAG system, including:

Why documents need to be chunked
How embeddings represent semantic meaning
How vector similarity search works
Why keyword retrieval can complement semantic search
Why reranking can improve retrieved context
How conversational queries can be rewritten for retrieval
How retrieval quality affects LLM answer quality
Why grounding an LLM in retrieved context is important for document-based applications
Future Direction

This project is intended to evolve beyond a basic RAG pipeline toward more capable AI systems.

## The planned progression:

V1 → RAG Backend
V2 → Usable Application
V3 → Automation & Integrations
V4 → Tool-Using AI Agent
V5 → Controlled Multi-Agent System
V6 → AI Business Operations Platform


# Author :
Muhammad Yaseen
