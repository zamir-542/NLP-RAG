```markdown
# Local RAG Assistant

General-purpose local RAG (Retrieval-Augmented Generation) system. Ingest any PDFs/docs → query in natural language → grounded, source-cited answers. Fully offline.

Demo dataset: semiconductor fab equipment manual (Edwards nXDS pump) — swap `data/` for any domain.

## Stack
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Vector store: ChromaDB (local, persistent)
- LLM: Ollama (llama3.2)
- PDF parsing: pdfplumber (table-aware)
- UI: Streamlit, multi-turn chat

## Why local
No cloud API calls. Applicable to any confidential-document use case — fab manuals, internal SOPs, legal docs, medical records.

## Setup
```bash
pip install -r requirements.txt
ollama pull llama3.2
```

## Usage
```bash
# drop any PDFs into data/
python ingest.py
streamlit run app.py
```

## Re-ingest
```bash
rm -rf vectordb   # or rmdir /s /q vectordb on Windows
python ingest.py
```

## Architecture
```
PDF → pdfplumber (text + tables) → chunk (800/150 overlap) → MiniLM embed → ChromaDB
Query → embed → top-k retrieve → context + history → llama3.2 → grounded answer
```

## Config (query.py)
| Param | Value |
|---|---|
| MODEL | llama3.2 |
| TOP_K | 5 |
| CHARS_PER_CHUNK | 900 |
| MAX_HISTORY | 6 |

## Grounding
Answers restricted to retrieved context + conversation transcript. No source → no answer. Prevents hallucination regardless of domain.

## Example use case (included)
Edwards nXDS Scroll Pump manual (public, fab support equipment) — demonstrates domain-specific technical Q&A: specs, maintenance intervals, fault codes.

## Data
`data/` + `vectordb/` gitignored — bring your own docs, none ship with repo.

## Limitations
- Table extraction quality depends on PDF structure; scanned docs need OCR (not implemented)
- CPU inference: ~5-20s/query; GPU drops to ~2-4s
```
