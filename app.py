import streamlit as st
from query import answer, _get_embedder

st.set_page_config(page_title="Fab Maintenance RAG", page_icon="🔧")
st.title("🔧 Fab Equipment Maintenance Assistant")
st.caption("Local RAG · Ollama · runs fully on-device")

@st.cache_resource
def _warm():
    _get_embedder()   # load embedder once at startup
    return True
_warm()

if "history" not in st.session_state:
    st.session_state.history = []   # list of (role, content) tuples

# Render conversation history
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

# Chat input (stays active after each message — no clearing needed)
if q := st.chat_input("Ask about the equipment manual..."):
    # Show user message
    with st.chat_message("user"):
        st.markdown(q)

    # Get answer with history for multi-turn memory
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ans, srcs = answer(q, st.session_state.history)
        st.markdown(ans)
        if srcs:
            st.caption("📄 Sources: " + ", ".join(srcs))

    # Save to history
    st.session_state.history.append(("user", q))
    st.session_state.history.append(("assistant", ans))

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    if st.session_state.history:
        if st.button("🗑️ Clear chat"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("Start chatting to see history here.")
