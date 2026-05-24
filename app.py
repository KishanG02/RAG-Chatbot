import streamlit as st
import os
import tempfile
from ingest import load_documents, chunk_documents
from rag_pipeline import build_vectorstore, build_rag_chain, query_rag

st.set_page_config(page_title="Multi-Doc RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 Multi-Document RAG Chatbot")
st.caption("Upload PDFs and ask questions about them using AI")

with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Ingest Documents", disabled=not uploaded_files):
        with tempfile.TemporaryDirectory() as tmpdir:
            with st.spinner("Loading & chunking PDFs..."):
                for f in uploaded_files:
                    with open(os.path.join(tmpdir, f.name), "wb") as out:
                        out.write(f.read())
                docs = load_documents(tmpdir)
                chunks = chunk_documents(docs)

            with st.spinner("Building vector store..."):
                vectorstore = build_vectorstore(chunks, persist=False)
                st.session_state.chain = build_rag_chain(vectorstore)

        st.success(f"✅ Ingested {len(chunks)} chunks from {len(uploaded_files)} PDF(s)")
        st.session_state.messages = []

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.info("👈 Upload PDFs in the sidebar to get started")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.caption(s)

if prompt := st.chat_input("Ask a question about your documents..."):
    if "chain" not in st.session_state:
        st.error("Please upload and ingest documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
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