# Fab Maintenance RAG

Local retrieval-augmented Q&A over fab equipment manuals and maintenance logs.
Runs **fully offline** via Ollama — no proprietary data ever leaves the machine.

## Features

- 📄 **PDF ingestion with table extraction** — `pdfplumber` extracts both prose and tables (maintenance schedules, spec sheets, torque values)
- 🧠 **Multi-turn chat** — Streamlit session history passed to the LLM so follow-up questions resolve pronouns and context correctly
- 🔍 **Context-aware retrieval** — last user turn is prepended to the embedding query for better pronoun/coreference resolution
- 🔒 **100% local** — Ollama LLM + ChromaDB vector store + sentence-transformers embeddings, no cloud APIs

## Requirements

- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- Python 3.10+

## Setup

```bash
# 1. Pull the LLM (one-time download, runs locally)
ollama pull llama3.2

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Drop your PDFs / text logs into data/
#    (folder is gitignored — proprietary docs stay local)

# 4. Ingest documents into the vector store
python ingest.py

# 5. Launch the chat UI
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## Re-ingesting after adding new documents

```powershell
# Windows PowerShell — stop the app first, then:
Remove-Item -Recurse -Force vectordb
python ingest.py
```

## Project layout

```
fab-maintenance-rag/
├── ingest.py       # Load → chunk → embed → upsert into ChromaDB
├── query.py        # Embed question → retrieve chunks → LLM answer
├── app.py          # Streamlit multi-turn chat UI
├── requirements.txt
├── data/           # Source documents (gitignored)
└── vectordb/       # Persisted ChromaDB index (gitignored)
```

## Configuration

Key constants in `query.py`:

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `llama3.2` | Ollama model name (`llama3.2:1b` for faster CPU inference) |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `MAX_TOKENS` | `300` | Max LLM output tokens |
| `CHARS_PER_CHUNK` | `900` | Characters of each chunk fed to the LLM |
| `MAX_HISTORY` | `6` | Conversation turns kept in context |

## Safety notes

- **Grounded-only prompt**: the LLM is instructed to answer only from retrieved context and say so explicitly when the answer isn't present — prevents hallucinated maintenance procedures.
- `data/` and `vectordb/` are gitignored so proprietary fab docs never leak into a public repo.
- Inference and retrieval are fully local (Ollama + ChromaDB + sentence-transformers).
