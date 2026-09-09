import os
import re
import json
import numpy as np
import pdfplumber
import streamlit as st
import requests

# Set page configuration
st.set_page_config(
    page_title="KORVA - AI HR Assistant",
    page_icon="💼",
    layout="wide"
)

# Initialize session state for chats
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None


# ==========================================
# RAG ENGINE CLASS
# ==========================================
class RAGEngine:
    def __init__(self, doc_dir="docs"):
        self.doc_dir = doc_dir
        self.documents = []
        self.load_documents()

    def load_documents(self):
        """Loads PDFs from directory and splits into pages."""
        self.documents = []
        if not os.path.exists(self.doc_dir):
            os.makedirs(self.doc_dir)

        for file_name in os.listdir(self.doc_dir):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(self.doc_dir, file_name)
                try:
                    with pdfplumber.open(file_path) as pdf:
                        for page_num, page in enumerate(pdf.pages, start=1):
                            text = page.extract_text()
                            if text:
                                self.documents.append({
                                    "name": file_name,
                                    "page": page_num,
                                    "path": file_path,
                                    "content": text.strip()
                                })
                except Exception as e:
                    st.error(f"Error loading {file_name}: {e}")

    def query_llm_engine(self, prompt):
        """Tries local Ollama first; dynamically queries active Groq models if offline."""
        # 1. Try Local Ollama (Active when running locally)
        try:
            url = "http://localhost:11434/api/generate"
            payload = {"model": "gemma3", "prompt": prompt, "stream": False}
            response = requests.post(url, json=payload, timeout=3)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception:
            pass  # Local Ollama not reachable; falling back to Cloud API

        # 2. Fallback to Groq API with Dynamic Model Retrieval
        try:
            from groq import Groq
            api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
            if not api_key:
                return "Error: Local Ollama is offline and GROQ_API_KEY is missing in Streamlit Cloud Secrets."

            client = Groq(api_key=api_key)

            # Fetch active models directly from Groq API to avoid decommission errors
            try:
                available_models = [m.id for m in client.models.list().data]
            except Exception:
                available_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

            # Prioritize active high-performance models
            preferred_order = [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
            ]
            
            # Filter based on active API availability
            target_models = [m for m in preferred_order if m in available_models]
            if not target_models and available_models:
                target_models = available_models

            # Execute completion call
            last_error = None
            for model_id in target_models:
                try:
                    completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_id,
                        temperature=0.2,
                    )
                    return completion.choices[0].message.content.strip()
                except Exception as err:
                    last_error = f"Model '{model_id}' failed: {str(err)}"
                    continue

            return f"LLM Generation Error: {last_error}"

        except Exception as e:
            return f"LLM Generation Error: {str(e)}"

    def retrieve(self, query, top_k=3):
        """Simple keyword matching retriever across documents."""
        if not self.documents:
            return []

        query_words = set(re.findall(r'\w+', query.lower()))
        scored_docs = []

        for doc in self.documents:
            content_words = set(re.findall(r'\w+', doc["content"].lower()))
            overlap = len(query_words.intersection(content_words))
            scored_docs.append((overlap, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k] if score > 0]

    def ask(self, query):
        """Executes full RAG pipeline and conditionally handles sources."""
        top_docs = self.retrieve(query)
        
        if not top_docs:
            return {
                "answer": "I couldn't find specific documentation addressing your prompt in the knowledge base.",
                "sources": []
            }

        context_str = "\n\n".join([
            f"--- Document: {d['name']} (Page {d['page']}) ---\n{d['content']}"
            for d in top_docs
        ])

        prompt = f"""You are KORVA, an AI HR Assistant. Answer the user's question accurately and thoroughly using ONLY the provided internal documentation context below.

Rules:
1. Synthesize a direct, concise, and helpful answer.
2. If the user's query cannot be answered using ONLY the context provided below, state strictly: "I couldn't find specific documentation addressing your prompt in the knowledge base."
3. Do not assume or invent facts outside the provided documentation.

Documentation Context:
{context_str}

User Question: {query}
Answer:"""

        # Unified LLM Call
        answer = self.query_llm_engine(prompt)

        # Fallback detection to prevent displaying sources when an answer wasn't generated
        fallback_phrases = [
            "couldn't find specific documentation",
            "llm generation error",
            "error:",
            "i don't have information"
        ]
        
        is_fallback = any(phrase in answer.lower() for phrase in fallback_phrases)

        # Conditionally return sources
        if is_fallback:
            sources = []
        else:
            sources = [{"name": doc["name"], "page": doc["page"], "path": doc["path"]} for doc in top_docs]

        return {"answer": answer, "sources": sources}


# Initialize RAG Engine
@st.cache_resource
def get_rag_engine():
    return RAGEngine()

rag_engine = get_rag_engine()


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.title("💼 KORVA HR")
    st.caption("Internal AI Assistant")

    if st.button("➕ New Chat", use_container_width=True):
        new_id = f"chat_{len(st.session_state.chats) + 1}"
        st.session_state.chats[new_id] = []
        st.session_state.active_chat = new_id
        st.rerun()

    st.divider()
    st.subheader("Recent Conversations")

    chat_ids = list(st.session_state.chats.keys())
    if chat_ids:
        if not st.session_state.active_chat:
            st.session_state.active_chat = chat_ids[-1]

        for c_id in reversed(chat_ids):
            msgs = st.session_state.chats[c_id]
            label = msgs[0]["content"][:25] + "..." if msgs else f"Conversation ({c_id})"
            
            if st.button(label, key=f"btn_{c_id}", use_container_width=True):
                st.session_state.active_chat = c_id
                st.rerun()
    else:
        st.info("No active chats yet. Start asking a question!")


# Ensure active chat exists
if not st.session_state.active_chat:
    new_id = "chat_1"
    st.session_state.chats[new_id] = []
    st.session_state.active_chat = new_id


# ==========================================
# CHAT INTERFACE & DISPLAY
# ==========================================
st.title("KORVA Workspace")
st.caption("Ask questions about internal policies, benefits, guidelines, and compliance.")

active_messages = st.session_state.chats.get(st.session_state.active_chat, [])

for msg in active_messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.container(border=True):
            st.markdown('<div class="assistant-label"><b>KORVA</b></div>', unsafe_allow_html=True)
            st.markdown(msg["content"])
            
            # Conditionally display sources only when available
            sources = msg.get("sources", [])
            if sources:
                st.divider()
                st.caption("Sources")
                for src in sources:
                    st.markdown(f"📄 `{src['name']}` *(Page {src['page']})*")


# ==========================================
# INPUT PROMPT HANDLING
# ==========================================
if user_input := st.chat_input("Ask KORVA about HR policies..."):
    # 1. Append User Message
    st.session_state.chats[st.session_state.active_chat].append({
        "role": "user",
        "content": user_input
    })

    # 2. Query RAG Engine
    res = rag_engine.ask(user_input)

    # 3. Append Assistant Response with Sources
    st.session_state.chats[st.session_state.active_chat].append({
        "role": "assistant",
        "content": res["answer"],
        "sources": res.get("sources", [])
    })

    # 4. Rerun to Render UI
    st.rerun()
