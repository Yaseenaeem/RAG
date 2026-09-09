import os
import sys
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

            # A. Fetch active models directly from Groq API to avoid decommission errors
            try:
                available_models = [m.id for m in client.models.list().data]
            except Exception:
                available_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

            # B. Prioritize active high-performance models
            preferred_order = [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
            ]
            
            # Filter based on active API availability
            target_models = [m for m in preferred_order if m in available_models]
            if not target_models and available_models:
                target_models = available_models

            # C. Execute completion call
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

    def ask(self, query, previous_questions=None):
        # A. FAISS Vector Retrieval (k=5)
        query_vector = self.model.encode([query]).astype("float32")
        distances, faiss_indices = self.index.search(query_vector, k=min(5, len(self.documents)))
        
        # B. BM25 Lexical Retrieval (k=5)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:min(5, len(self.documents))]
        
        # C. Combined Fusion & Re-ranking
        candidate_indices = list(set(list(faiss_indices[0]) + list(bm25_indices)))
        scored_candidates = []
        for idx in candidate_indices:
            bm25_score = bm25_scores[idx]
            vec_dist = distances[0][0] if idx in faiss_indices[0] else 2.0
            combined_score = bm25_score + (1.0 / (1.0 + vec_dist))
            scored_candidates.append((combined_score, self.documents[idx]))
            
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for score, doc in scored_candidates[:3]]
        
        context_str = "\n\n".join([f"Document [{doc['name']}]:\n{doc['text']}" for doc in top_docs])
        
        # D. System Prompt
        prompt = f"""You are KORVA, an AI HR Assistant. Answer the user's question accurately and thoroughly using ONLY the provided internal documentation context below.

Rules:
1. Synthesize a direct, concise, and helpful answer.
2. If the user's query cannot be answered using ONLY the context provided below, state strictly: "I couldn't find specific documentation addressing your prompt in the knowledge base."
3. Do not assume or invent facts outside the provided documentation.

Documentation Context:
{context_str}

User Question: {query}
Answer:"""

        # E. Unified LLM Call
        answer = self.query_llm_engine(prompt)

        # Detect fallback or error responses
        fallback_phrases = [
            "couldn't find specific documentation",
            "llm generation error",
            "error:",
            "i don't have information"
        ]
        is_fallback = any(phrase in answer.lower() for phrase in fallback_phrases)

        # Only return sources if the answer is NOT a fallback and top search score is sufficient
        if is_fallback or (scored_candidates and scored_candidates[0][0] < 0.1):
            sources = []
        else:
            sources = [{"name": doc["name"], "page": doc["page"], "path": doc["path"]} for doc in top_docs]
        
        return {"answer": answer, "sources": sources}
        
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
