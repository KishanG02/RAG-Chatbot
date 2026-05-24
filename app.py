import streamlit as st
import os
import tempfile
from ingest import load_documents, chunk_documents
from rag_pipeline import build_vectorstore, build_rag_chain, query_rag

st.set_page_config(page_title="Multi-Doc RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 Multi-Document RAG Chatbot")
st.caption("Upload PDFs and ask questions about them using AI")

# Clear everything if no documents are loaded (fresh tab/session)
if "initialized" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chain = None
    st.session_state.uploaded_filenames = []
    st.session_state.initialized = True

with st.sidebar:
    st.header("📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Ingest Documents", disabled=not uploaded_files):
        # Reset everything on new ingestion
        st.session_state.messages = []
        st.session_state.chain = None
        st.session_state.uploaded_filenames = []

        with tempfile.TemporaryDirectory() as tmpdir:
            with st.spinner("Loading & chunking PDFs..."):
                for f in uploaded_files:
                    filepath = os.path.join(tmpdir, f.name)
                    with open(filepath, "wb") as out:
                        out.write(f.read())
                    st.session_state.uploaded_filenames.append(f.name)

                docs = load_documents(tmpdir)
                chunks = chunk_documents(docs)

            with st.spinner("Building vector store..."):
                vectorstore = build_vectorstore(chunks, persist=False)
                st.session_state.chain = build_rag_chain(vectorstore)

        st.success(f"✅ Ingested {len(chunks)} chunks from {len(uploaded_files)} PDF(s)")

    # Show loaded documents
    if st.session_state.get("uploaded_filenames"):
        st.divider()
        st.markdown("**📄 Loaded documents:**")
        for name in st.session_state.uploaded_filenames:
            st.caption(f"• {name}")

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        if st.session_state.chain:
            st.session_state.chain.memory.clear()
        st.rerun()

    if st.button("🔄 Reset Everything"):
        st.session_state.messages = []
        st.session_state.chain = None
        st.session_state.uploaded_filenames = []
        st.rerun()

# Main chat area
if not st.session_state.chain:
    st.info("👈 Upload PDFs in the sidebar and click **Ingest Documents** to get started")
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Sources"):
                    for s in msg["sources"]:
                        st.caption(s)

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                answer, sources = query_rag(st.session_state.chain, prompt)
            st.markdown(answer)
            if sources:
                with st.expander("📄 Sources"):
                    for s in sources:
                        st.caption(s)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
