import streamlit as st
from main import run_pipeline
from generate import rewrite_query

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="🔥",
    layout="centered"
)

# ---------- WARM AESTHETIC CSS ----------
st.markdown("""
<style>
    .stApp {
        background-color: #FAF6F1;
    }

    h1, [data-testid="stMarkdownContainer"] h1 {
        color: #C6592F !important;
        font-weight: 700 !important;
    }

    .subtitle {
        color: #8A7B6C !important;
        font-size: 15px;
        margin-bottom: 30px;
    }

    .chat-bubble-user {
        background-color: #C6592F;
        color: white !important;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 15px;
    }

    .chat-bubble-assistant {
        background-color: #FFFFFF;
        color: #3D3229 !important;
        padding: 14px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        border: 1px solid #EDE3D8;
        font-size: 15px;
        line-height: 1.6;
    }

    .context-box {
        background-color: #FFF9F2;
        border-left: 3px solid #E8A87C;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 13px;
        color: #8A7B6C !important;
        margin: 6px 0;
    }

    [data-testid="stChatInput"] {
        background-color: #FFF9F2 !important;
        border-top: 1px solid #EDE3D8;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #FFFFFF !important;
        color: #3D3229 !important;
    }

    .stExpander {
        background-color: #FFFFFF !important;
        border: 1px solid #EDE3D8 !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1>🔥 Hybrid RAG Assistant</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Ask questions grounded in your documents — powered by dense + keyword search, fused together.</div>",
    unsafe_allow_html=True
)

# ---------- SESSION STATE (keeps chat history) ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- DISPLAY CHAT HISTORY ----------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-bubble-user'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-assistant'>{msg['content']}</div>", unsafe_allow_html=True)

        if "chunks" in msg:
            with st.expander("📄 View retrieved context"):
                for i, chunk in enumerate(msg["chunks"]):
                    st.markdown(f"<div class='context-box'><b>Chunk {i+1}:</b> {chunk}</div>", unsafe_allow_html=True)

# ---------- INPUT BOX ----------
query = st.chat_input("Ask something about your documents...")

if query:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f"<div class='chat-bubble-user'>{query}</div>", unsafe_allow_html=True)

    # Rewrite vague follow-ups into standalone questions using chat history
    with st.spinner("Understanding your question..."):
        standalone_query = rewrite_query(query, st.session_state.messages[:-1])

    # Run the pipeline using the REWRITTEN query
    with st.spinner("Searching and thinking..."):
        answer, used_chunks = run_pipeline(standalone_query)

    # Show assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": used_chunks
    })
    st.markdown(f"<div class='chat-bubble-assistant'>{answer}</div>", unsafe_allow_html=True)

    with st.expander("📄 View retrieved context"):
        for i, chunk in enumerate(used_chunks):
            st.markdown(f"<div class='context-box'><b>Chunk {i+1}:</b> {chunk}</div>", unsafe_allow_html=True)