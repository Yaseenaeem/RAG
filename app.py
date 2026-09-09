import os
import sys
import html
import numpy as np
import requests
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="KORVA Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. CSS Override (Unified Dark Blue Canvas + Custom Source Cards)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* A. UNIFIED DARK BLUE CANVAS */
    .stApp, 
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"],
    [data-testid="stSidebar"], 
    [data-testid="stSidebarContent"],
    [data-testid="stBottom"],
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stBottom"] > div {
        background-color: #0A192F !important;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #1E293B !important;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 850px;
    }

    /* B. GLOBAL WHITE TEXT ENFORCEMENT */
    h1, h2, h3, h4, h5, h6, p, span, div, label, caption,
    [data-testid="stSidebar"] *, 
    .main * {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }

    /* NEON PURPLE SIDEBAR BRANDING */
    .brand-title {
        color: #A855F7 !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        letter-spacing: 1px;
        margin-bottom: 0px;
        text-shadow: 0 0 10px rgba(168, 85, 247, 0.4);
    }
    .brand-subtitle {
        color: #94A3B8 !important;
        font-size: 0.82rem !important;
        margin-bottom: 1.5rem;
    }

    /* C. SIDEBAR ELEMENTS & RED BUTTON */
    [data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        color: #FFFFFF !important;
        background-color: #DC2626 !important;
        border: 1px solid #EF4444 !important;
        font-weight: 700 !important;
        transition: all 0.2s ease-in-out !important;
    }
    [data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {
        background-color: #B91C1C !important;
        border-color: #10B981 !important;
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] div.stButton > button:not([kind="primary"]) {
        color: #FFFFFF !important;
        background-color: #112240 !important;
        border: 1px solid #233554 !important;
        font-weight: 500 !important;
        transition: all 0.2s ease-in-out !important;
    }
    [data-testid="stSidebar"] div.stButton > button:not([kind="primary"]):hover {
        border-color: #10B981 !important;
        color: #FFFFFF !important;
    }

    /* D. CHAT CARDS & GREEN HOVER BORDER */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #112240 !important;
        border: 1px solid #233554 !important;
        border-radius: 8px !important;
        transition: border-color 0.2s ease-in-out !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #10B981 !important;
    }

    .user-label {
        font-size: 0.85rem;
        font-weight: 700;
        color: #38BDF8 !important;
        margin-bottom: 4px;
    }
    .assistant-label {
        font-size: 0.85rem;
        font-weight: 700;
        color: #10B981 !important;
        margin-bottom: 4px;
    }

    /* E. FAQ CARDS & BUTTONS */
    div[data-testid="stColumn"] div.stButton > button {
        padding: 6px 12px !important;
        min-height: 38px !important;
        font-size: 0.9rem !important;
        color: #FFFFFF !important;
        background-color: #112240 !important;
        border: 1px solid #233554 !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stColumn"] div.stButton > button:hover {
        border-color: #10B981 !important;
        color: #FFFFFF !important;
    }

    /* F. CUSTOM SOURCE CARDS */
    .source-card {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        padding: 8px 14px !important;
        margin-bottom: 8px !important;
        color: #FFFFFF !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        transition: border-color 0.2s ease-in-out !important;
    }
    .source-card:hover {
        border-color: #10B981 !important;
        background-color: #1E293B !important;
        color: #FFFFFF !important;
    }

    /* G. BOTTOM CHAT INPUT FIELD */
    [data-testid="stChatInput"] textarea {
        color: #FFFFFF !important;
        background-color: #112240 !important;
        border: 1px solid #233554 !important;
        border-radius: 8px !important;
        outline: none !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #94A3B8 !important;
    }
    [data-testid="stChatInput"] textarea:focus,
    [data-testid="stChatInput"] textarea:focus-visible,
    [data-testid="stChatInput"] > div:focus-within {
        border-color: #10B981 !important;
        outline: none !important;
        box-shadow: none !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 3. Search Engine Dependencies & RAGEngine
# -----------------------------------------------------------------------------
class RAGEngine:
    def __init__(self):
        # 1. Complete Knowledge Base Setup
        self.documents = [
            {
                "name": "Remote_Work_Policy.pdf",
                "page": 1,
                "path": "docs/Remote_Work_Policy.pdf",
                "text": "Remote Work Eligibility: All full-time employees with at least 3 months of tenure are eligible to work remotely up to 2 days per week. Core working hours are 10 AM to 4 PM EST."
            },
            {
                "name": "Employee_Benefits.pdf",
                "page": 4,
                "path": "docs/Employee_Benefits.pdf",
                "text": "Benefits Overview: The company offers comprehensive medical, dental, and vision insurance. Employees receive 20 annual paid leave days plus corporate holidays."
            },
            {
                "name": "Dress_Code_Policy.pdf",
                "page": 2,
                "path": "docs/Dress_Code_Policy.pdf",
                "text": "Dress Code Policy: Standard office attire is business casual from Monday to Thursday (professional slacks, trousers, blouses, collared shirts). Smart casual wear (including clean denim) is permitted on Fridays."
            },
            {
                "name": "IT_Security_Guidelines.pdf",
                "page": 3,
                "path": "docs/IT_Security_Guidelines.pdf",
                "text": "IT Security Rules: Passwords must be updated every 90 days. Multi-factor authentication (MFA) is required on all company accounts and software portals."
            },
            {
                "name": "Drug_Policy.pdf",
                "page": 1,
                "path": "docs/Drug_Policy.pdf",
                "text": "Drug Policy: The company maintains a zero-tolerance drug-free workplace environment. Possession, consumption, or distribution of illegal substances (including marijuana and cocaine) on company premises or during work hours is strictly prohibited. Any violation of this policy will result in immediate termination of employment."
            }
        ]
        self.document_count = len(self.documents)
        
        # 2. Vector Indexing
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        corpus_texts = [doc["text"] for doc in self.documents]
        embeddings = self.model.encode(corpus_texts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype("float32"))
        
        # 3. BM25 Lexical Indexing
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def query_llm_engine(self, prompt):
        try:
            # GROQ API EXECUTION
            if os.getenv("GROQ_API_KEY"):
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2
                    }
                )
                data = response.json()
                # Parse OpenAI/Groq response schema
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"].strip()

            # OLLAMA LOCAL EXECUTION (Fallback if no GROQ_API_KEY)
            else:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3",
                        "prompt": prompt,
                        "stream": False
                    }
                )
                data = response.json()
                # Parse Ollama response schema
                if "response" in data:
                    return data["response"].strip()

            return "I couldn't find specific documentation addressing your prompt in the knowledge base."

        except Exception as e:
            print(f"LLM Engine Error: {e}")
            return "I couldn't find specific documentation addressing your prompt in the knowledge base."

    def ask(self, query, previous_questions=None):
        # 1. FAISS Vector Search
        query_vector = self.model.encode([query]).astype("float32")
        distances, faiss_indices = self.index.search(query_vector, k=min(3, len(self.documents)))
        
        # Simple distance-to-similarity conversion
        vec_scores = {
            idx: float(dist) 
            for idx, dist in zip(faiss_indices[0], distances[0])
        }

        # 2. BM25 Lexical Search
        tokenized_query = query.lower().split()
        bm25_raw = self.bm25.get_scores(tokenized_query)

        # 3. Hybrid Ranking
        scored_candidates = []
        for idx in faiss_indices[0]:
            if idx < len(self.documents):
                doc = self.documents[idx]
                scored_candidates.append(doc)

        # Always take the top retrieved documents (up to 3)
        top_docs = scored_candidates[:3]

        if not top_docs:
            return {
                "answer": "I couldn't find specific documentation addressing your prompt in the knowledge base.",
                "sources": []
            }

        # 4. Context Assembly
        context_str = "\n\n".join([f"Document [{doc['name']}]:\n{doc['text']}" for doc in top_docs])

        # 5. System Prompt - Instructs LLM to evaluate relevance directly
        prompt = f"""You are KORVA, an AI HR Assistant. Answer the user's question accurately using ONLY the documentation context below.

Rules:
1. Synthesize a direct answer using the provided context.
2. Do NOT include conversational sign-offs or general pleasantries.
3. If the context does not contain the answer to the user's question, output EXACTLY: "I couldn't find specific documentation addressing your prompt in the knowledge base."

Documentation Context:
{context_str}

User Question: {query}
Answer:"""

        # 6. Call LLM safely
        answer_text = self.query_llm_engine(prompt)

        # 7. Match sources: If LLM gives fallback, don't show sources; otherwise show the top document
        if "couldn't find specific documentation" in answer_text.lower():
            sources = []
        else:
            # Attach ONLY the #1 best matching document to avoid cluttering unrelated source cards
            top_source = top_docs[0]
            sources = [{"name": top_source["name"], "page": top_source["page"], "path": top_source["path"]}]

        return {"answer": answer_text, "sources": sources}
        
# Initialize Engine
@st.cache_resource
def get_rag_engine():
    return RAGEngine()

try:
    engine = get_rag_engine()
except Exception as e:
    st.error(f"Error loading RAG Engine: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 4. State Management
# -----------------------------------------------------------------------------
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

def submit_query(query_text):
    if not query_text:
        return
    
    if not st.session_state.active_chat or st.session_state.active_chat not in st.session_state.chats:
        chat_title = query_text.strip().capitalize()
        if len(chat_title) > 28:
            chat_title = chat_title[:26] + "..."
        st.session_state.chats[chat_title] = []
        st.session_state.active_chat = chat_title

    messages = st.session_state.chats[st.session_state.active_chat]
    messages.append({"role": "user", "content": query_text})
    
    prev_queries = [m["content"] for m in messages if m["role"] == "user"][:-1]
    
    with st.spinner("Generating answer..."):
        res = engine.ask(query_text, previous_questions=prev_queries)
        
    messages.append({
        "role": "assistant",
        "content": res["answer"],
        "sources": res.get("sources", [])
    })

# -----------------------------------------------------------------------------
# 5. Modals / Dialogs
# -----------------------------------------------------------------------------
@st.dialog("About Assistant")
def open_about_modal():
    st.subheader("💬 KORVA Assistant")
    st.write("An enterprise document retrieval assistant powered by RAG.")
    st.markdown("""
    * **LLM Engine:** Local Embedding + Lexical Ranker
    * **Search:** FAISS Hybrid Vector Search
    * **Security:** Enterprise RAG Architecture
    """)

# -----------------------------------------------------------------------------
# 6. Sidebar Layout
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand-title">KORVA</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Knowledge & Organizational Retrieval Assistant</div>', unsafe_allow_html=True)

    if st.button("＋ New conversation", type="primary", use_container_width=True):
        st.session_state.active_chat = None
        st.session_state.pending_prompt = None
        st.rerun()

    st.write("")
    
    if st.session_state.chats:
        st.caption("RECENT CONVERSATIONS")
        for chat_title in list(st.session_state.chats.keys()):
            is_active = chat_title == st.session_state.active_chat
            label = f"💬 {chat_title}" if is_active else f"📄 {chat_title}"
            
            if st.button(label, key=f"nav_{chat_title}", use_container_width=True):
                st.session_state.active_chat = chat_title
                st.session_state.pending_prompt = None
                st.rerun()

    st.divider()

    st.caption("KNOWLEDGE BASE")
    st.write(f"📚 **{engine.document_count} documents indexed**")

    st.write("")
    if st.button("ⓘ About", use_container_width=True):
        open_about_modal()

if st.session_state.pending_prompt:
    prompt_to_run = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    submit_query(prompt_to_run)
    st.rerun()

# -----------------------------------------------------------------------------
# 7. Main Workspace Container
# -----------------------------------------------------------------------------
workspace = st.empty()

with workspace.container():
    active_chat = st.session_state.active_chat
    messages = st.session_state.chats.get(active_chat, []) if active_chat else []

    if not active_chat or not messages:
        st.markdown("<h2 style='text-align: center; margin-top: 2rem; color: #FFFFFF;'>How can KORVA help you today?</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94A3B8; margin-bottom: 2rem;'>Ask anything about internal policies, guidelines, and corporate documentation.</p>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.caption("📋 **Remote Work**")
            if st.button("Explain remote work policy eligibility", key="faq_1", use_container_width=True):
                st.session_state.pending_prompt = "Explain complete remote work policy"
                st.rerun()

            st.write("")
            st.caption("👔 **Dress Code**")
            if st.button("What is the company dress code?", key="faq_2", use_container_width=True):
                st.session_state.pending_prompt = "What is the company dress code policy?"
                st.rerun()

        with c2:
            st.caption("💡 **Benefits**")
            if st.button("Summarize key employee benefits", key="faq_3", use_container_width=True):
                st.session_state.pending_prompt = "Summarize the key employee benefits and leave policies"
                st.rerun()

            st.write("")
            st.caption("🔒 **IT Security**")
            if st.button("Explain IT security and data guidelines", key="faq_4", use_container_width=True):
                st.session_state.pending_prompt = "Explain IT security and data protection policies"
                st.rerun()

    else:
        for m_idx, msg in enumerate(messages):
            if msg["role"] == "user":
                with st.container(border=True):
                    st.markdown('<div class="user-label">You</div>', unsafe_allow_html=True)
                    st.write(msg["content"])
            else:
                with st.container(border=True):
                    st.markdown('<div class="assistant-label">KORVA</div>', unsafe_allow_html=True)
                    st.markdown(msg["content"])
                    
                    # Display sources ONLY if valid sources exist and the answer isn't a fallback message
                    sources = msg.get("sources", [])
                    fallback_phrases = [
                        "couldn't find specific documentation",
                        "llm generation error",
                        "error:"
                    ]
                    is_fallback_msg = any(phrase in msg["content"].lower() for phrase in fallback_phrases)

                    if sources and not is_fallback_msg:
                        st.divider()
                        st.caption("Sources")
                        for src in sources:
                            safe_name = html.escape(src['name'])
                            card_html = f"""
                            <div class="source-card">
                                📄 <span>{safe_name} (Page {src['page']})</span>
                            </div>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)

st.write("")
st.caption("🔒 Responses are generated from your organization's documents.")

if prompt := st.chat_input("Ask anything about your knowledge base..."):
    submit_query(prompt)
    st.rerun()
