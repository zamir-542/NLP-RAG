```markdown
# Fab Maintenance RAG

Local RAG assistant for semiconductor/fab equipment manuals. Fully offline — no cloud APIs, no data leaves the machine.

## Stack
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector store**: ChromaDB (persistent, local)
- **LLM**: Ollama (llama3.2)
- **PDF parsing**: pdfplumber (table-aware extraction)
- **UI**: Streamlit, multi-turn chat

## Why local
Fab documentation is often confidential. This architecture proves the RAG pattern works with zero external API calls — relevant for semiconductor/industrial environments with data residency constraints.

## Setup
```bash
pip install -r requirements.txt
ollama pull llama3.2
```

## Usage
```bash
# 1. Drop PDFs/docs into data/
# 2. Ingest
python ingest.py

# 3. Run
streamlit run app.py
```

## Re-ingest (after changing chunking/extraction logic)
```bash
rmdir /s /q vectordb   # Windows
rm -rf vectordb         # macOS/Linux
python ingest.py
```

## Architecture
```
PDF → pdfplumber (text + tables) → chunk (800/150 overlap)
    → MiniLM embed → ChromaDB
Query → embed → top-k retrieve → context + history → llama3.2 → grounded answer
```

## Config (query.py)
| Param | Value | Notes |
|---|---|---|
| MODEL | llama3.2 | swap to `llama3.2:1b` for CPU speed, weaker grounding |
| TOP_K | 5 | chunks retrieved per query |
| CHARS_PER_CHUNK | 900 | context trimmed per chunk |
| MAX_HISTORY | 6 | turns kept for multi-turn context |

## Grounding
System prompt restricts answers to retrieved context + conversation transcript only. No answer without a source citation — prevents hallucinated maintenance procedures, a real safety concern for equipment servicing.

## Data
`data/` and `vectordb/` are gitignored — no proprietary docs ship with the repo. Demo uses a public Edwards nXDS Scroll Pump manual (industrial vacuum equipment used in fab support systems).

## Limitations
- Table extraction depends on PDF structure quality; scanned/image PDFs need OCR (not implemented)
- CPU inference: ~5-20s/query. GPU (CUDA) drops this to ~2-4s.
```
