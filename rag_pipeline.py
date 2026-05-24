from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_PERSIST_DIR = "chroma_db"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def build_vectorstore(chunks, persist=False):
    embeddings = get_embeddings()
    if persist:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name="rag_docs"
        )
        vectorstore.persist()
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="rag_docs"
        )
    print(f"Stored {len(chunks)} vectors in ChromaDB")
    return vectorstore

def load_vectorstore():
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="rag_docs"
    )
    return vectorstore

def build_rag_chain(vectorstore):
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    return chain

def query_rag(chain, question: str):
    result = chain({"question": question})
    answer = result["answer"]

    sources = []
    seen = set()
    for doc in result["source_documents"]:
        meta = doc.metadata
        key = f"{meta.get('source', 'Unknown')} — p.{meta.get('page', '?')}"
        if key not in seen:
            sources.append(key)
            seen.add(key)

    return answer, sources
