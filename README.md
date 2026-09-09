# KORVA — Enterprise RAG Assistant (V2)

A local Retrieval-Augmented Generation (RAG) system with a custom Streamlit user interface, hybrid retrieval pipeline, multi-session chat history, and document citation tracking.

This project updates the previous terminal-based backend (**V1**) into an interactive web application (**V2**).

---

## Live Application

Try the deployed application directly in your browser:  
[Insert Your Deployed Streamlit Link Here]

No installation or setup required, click the link above to test KORVA directly in your browser.

---

## Overview

The goal of V2 is to transition the backend RAG pipeline built in V1 into a usable interface. The system allows users to interact with indexed PDF documents using conversational search, pre-built FAQ options, multi-thread session management, and explicit source page citations.

---

## Key Features (V2 Updates)

* **Interactive Interface:** Custom dark-themed Streamlit UI designed for multi-turn document retrieval.
* **Hybrid Search Engine:** Combines dense vector search (FAISS) with keyword matching (BM25).
* **Source & Page Attribution:** Displays specific document names and page numbers associated with generated responses.
* **Session Management:** Sidebar navigation to start new conversations or return to previous chat threads.
* **Sample Knowledge Base:** Pre-indexed HR and corporate policy documents for testing.

---

## Architecture

```text
User Input / FAQ Selection
       ↓
Hybrid Candidate Retrieval (FAISS + BM25)
       ↓
Context Aggregation & Grounding
       ↓
Response Generation + Source Citations
```

---

## Technologies Used

* **Python**
* **Streamlit** (UI & Session State)
* **FAISS** (Vector Index)
* **BM25** (Keyword Retrieval)
* **Sentence Transformers** (`all-MiniLM-L6-v2`)
* **NumPy**
* **PyPDF**

---

## Repository Structure

```text
rag-assistant/
│
├── app.py              # Main application script (UI + RAG Engine)
├── requirements.txt   # Application dependencies
├── README.md           # Project documentation
├── .gitignore          # Git exclusion rules
├── docs/               # Sample HR policy PDF files
└── screenshots/        # Interface preview images
```
---

## Local Setup (Optional for Developers)
If you prefer to run the raw source code locally on your machine:

1. Clone the repository:

```Bash
git clone https://github.com/your-username/rag-assistant.git
```

2. Install dependencies:

```Bash
pip install -r requirements.txt
```

3. Run the application:

```Bash
streamlit run app.py
```
---


## Project Progression
* **V1: Backend RAG Pipeline (Hybrid Search, Reranking, Local LLM)**
* **V2: Web Application & UI (Streamlit, Session History, Citations)**
* **V3: Automations & Integrations**

---

## Author

Muhammad Yaseen Naeem
