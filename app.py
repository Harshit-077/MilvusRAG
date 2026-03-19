import streamlit as st
from main import chat, retrieve  # assuming main.py has `chat()` and `retrieve()`

st.set_page_config(page_title="Milvus RAG", layout="wide")

st.title("📚 Milvus RAG Chat")
st.markdown("Ask questions from your documents and see the context sources.")

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query_history" not in st.session_state:
    st.session_state.query_history = []

# ---------------- USER INPUT ----------------
query = st.text_input("🔍 Enter your question here")

if st.button("Send"):
    if query.strip():
        with st.spinner("Thinking..."):
            answer = chat(query)

        # Save chat
        st.session_state.query_history.append(query)
        st.session_state.chat_history.append(answer)

        # Display latest response
        st.subheader("🧠 Answer")
        st.write(answer)

        # Show retrieved sources
        sources = retrieve(query)
        if sources:
            st.subheader("📚 Sources / Context")
            for i, doc in enumerate(sources):
                st.markdown(f"**Source {i+1}: {doc['source']}**")
                st.write(doc["text"])
        else:
            st.info("No sources retrieved.")
    else:
        st.warning("Enter a question before sending!")

# ---------------- CHAT HISTORY ----------------
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 Chat History")
    for q, a in zip(st.session_state.query_history, st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")