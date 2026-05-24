import streamlit as st
import os
from ingest import load_documents, chunk_documents
from rag_pipeline import build_vectorstore, load_vectorstore, build_rag_chain, query_rag

st.set_page_config(page_title="Multi-Doc RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 Multi-Document RAG Chatbot")

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )

    if st.button("Ingest Documents", disabled=not uploaded_files):
        with st.spinner("Loading & chunking PDFs..."):
            os.makedirs("data", exist_ok=True)
            for f in uploaded_files:
                with open(f"data/{f.name}", "wb") as out:
                    out.write(f.read())
            docs = load_documents("data/")
            chunks = chunk_documents(docs)

        with st.spinner("Generating embeddings & storing in ChromaDB..."):
            vectorstore = build_vectorstore(chunks)
            st.session_state.chain = build_rag_chain(vectorstore)

        st.success(f"✅ Ingested {len(chunks)} chunks from {len(uploaded_files)} PDFs")

    if st.button("Load Existing DB"):
        vs = load_vectorstore()
        st.session_state.chain = build_rag_chain(vs)
        st.success("✅ Loaded existing ChromaDB")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.caption(s)

if prompt := st.chat_input("Ask a question about your documents..."):
    if "chain" not in st.session_state:
        st.error("Please ingest documents or load existing DB first.")
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