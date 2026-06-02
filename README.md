# 🤖 Multi-Document RAG Chatbot

A production-quality Retrieval-Augmented Generation (RAG) chatbot built with
**LangChain**, **ChromaDB**, and **OpenAI API** — featuring semantic search,
context-aware multi-turn Q&A, and source attribution across custom PDF corpora.

Live Demo: https://rag-chatbot-enrdbjdfrqgv8hpfbjtawz.streamlit.app/

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                   │
│                                                         │
│  PDFs  ──►  PyPDFLoader  ──►  RecursiveTextSplitter    │
│             (per page)        (1000 chars, 200 overlap) │
│                                     │                   │
│                                     ▼                   │
│                          OpenAI Embeddings              │
│                       (text-embedding-3-small)          │
│                                     │                   │
│                                     ▼                   │
│                            ChromaDB (local)             │
│                          persisted to ./vectorstore/    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  QUERY / CHAT PIPELINE                  │
│                                                         │
│  User Question                                          │
│       │                                                 │
│       ▼  (+ chat history)                               │
│  Condense Question  ──►  GPT-4o-mini                   │
│  (standalone form)                                      │
│       │                                                 │
│       ▼                                                 │
│  ChromaDB Retriever  (top-k semantic similarity)        │
│       │                                                 │
│       ▼                                                 │
│  QA Chain  ──►  GPT-4o-mini  ──►  Answer + Sources     │
│                                                         │
│  ConversationBufferWindowMemory (last 5 turns)          │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
rag_chatbot/
│
├── app.py               # Streamlit UI — entry point
├── ingest.py            # CLI ingestion script (optional)
├── requirements.txt     # Python dependencies
├── .env.example         # Copy to .env and add your API key
│
├── utils/
│   ├── __init__.py
│   ├── ingestor.py      # PDF loading, chunking, embedding, ChromaDB
│   └── rag_chain.py     # ConversationalRetrievalChain + source parsing
│
├── data/                # Drop your PDF files here
│   └── (your PDFs)
│
└── vectorstore/         # Auto-created by ChromaDB
    └── chroma.sqlite3
```

---

## ⚙️ Setup

### 1. Clone / create the project directory

```bash
mkdir rag_chatbot && cd rag_chatbot
# copy all files here
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

```bash
cp .env.example .env
# Open .env and replace "your_openai_api_key_here" with your real key
```

---

## 🚀 Running the App

### Option A — Streamlit UI (recommended)

```bash
streamlit run app.py
```

Then in the sidebar:
1. Paste your OpenAI API key (if not set in `.env`)
2. Upload one or more PDFs
3. Click **⚡ Build Index** (only needed once)
4. Start chatting!

On subsequent runs, click **🔄 Load Index** to reuse the stored vectors
without paying for re-embedding.

### Option B — CLI pre-ingest then run UI

```bash
# Put PDFs in ./data/ first, then:
python ingest.py --pdf_dir data/

# Now run the app and click "Load Index"
streamlit run app.py
```

---

## 💡 Key Concepts Explained

### PDF Chunking
Long PDFs are split into 1000-character overlapping chunks so:
- No single chunk exceeds the LLM's context window
- Related sentences at chunk boundaries aren't lost (200-char overlap)

### Embeddings
Each chunk is converted to a 1536-dimensional vector using OpenAI's
`text-embedding-3-small`. Similar meaning → vectors close in space.

### ChromaDB
A local vector database that stores chunks + their embeddings on disk.
At query time, it does cosine similarity search to find the top-k most
relevant chunks for your question.

### Conversational Memory
`ConversationBufferWindowMemory` keeps the last 5 exchanges. A "condense"
step rewrites follow-up questions (e.g., "What about page 3?") into
self-contained queries (e.g., "What does page 3 say about X?") before
hitting the retriever — this is what makes multi-turn chat work correctly.

### Source Attribution
Every retrieved chunk carries metadata (filename, page number). After
each answer, the app surfaces which documents and pages the answer
came from so you can verify.

---

## 🔧 Configuration Knobs

| Setting | Where | Default | Effect |
|---|---|---|---|
| `CHUNK_SIZE` | `utils/ingestor.py` | 1000 | Larger = more context per chunk |
| `CHUNK_OVERLAP` | `utils/ingestor.py` | 200 | Larger = fewer missed boundaries |
| `top_k` | Sidebar slider | 4 | More chunks = richer context, higher cost |
| `memory_window` | `utils/rag_chain.py` | 5 | More turns kept in memory |
| `model` | `utils/rag_chain.py` | gpt-4o-mini | Swap to gpt-4o for better quality |
| `temperature` | `utils/rag_chain.py` | 0.2 | Lower = more factual answers |

---

## 💰 Cost Estimate (OpenAI)

| Operation | Model | Approx cost |
|---|---|---|
| Embedding 100-page PDF | text-embedding-3-small | ~$0.002 |
| Each chat turn | gpt-4o-mini | ~$0.001–0.005 |

---

## 🧩 Possible Extensions

- **Hybrid search** — combine BM25 keyword search with semantic search
- **Re-ranking** — use a cross-encoder to re-rank retrieved chunks
- **Multi-modal** — add image extraction from PDFs using GPT-4o vision
- **Authentication** — add Streamlit login for multi-user deployments
- **Cloud deployment** — deploy to Streamlit Cloud or Hugging Face Spaces
- **Different LLMs** — swap OpenAI for Ollama (local) or Anthropic Claude
