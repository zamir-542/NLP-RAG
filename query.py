import chromadb, ollama
from pathlib import Path
from sentence_transformers import SentenceTransformer
from functools import lru_cache

SCRIPT_DIR = Path(__file__).parent
MODEL = "llama3.2"
TOP_K = 5
CHARS_PER_CHUNK = 900
MAX_TOKENS = 300
MAX_HISTORY = 6

@lru_cache(maxsize=1)
def _get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

col = chromadb.PersistentClient(str(SCRIPT_DIR / "vectordb")).get_collection("fab_docs")

def answer(question, history=None):
    history = history or []
    # embed raw question + last user turn for pronoun/context resolution
    last_user = next((c for r, c in reversed(history) if r == "user"), "")
    search_q = f"{last_user} {question}".strip() if history else question

    q_emb = _get_embedder().encode([search_q]).tolist()
    res = col.query(query_embeddings=q_emb, n_results=TOP_K)
    docs, metas = res["documents"][0], res["metadatas"][0]
    if not docs:
        return "No relevant information found in the manual.", []

    ctx = "\n\n".join(f"[{m['source']}]\n{d[:CHARS_PER_CHUNK]}" for d, m in zip(docs, metas))
    convo = "\n".join(f"{r}: {c}" for r, c in history[-MAX_HISTORY:])
    user_block = (f"Conversation so far:\n{convo}\n\n" if convo else "") + \
                 f"Context:\n{ctx}\n\nCurrent question: {question}"

    out = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content":
                "Answer ONLY from the context or conversation transcript. "
                "For questions about earlier questions, read the transcript literally. "
                "If not present, say so briefly. Be concise."},
            {"role": "user", "content": user_block},
        ],
        options={"num_predict": MAX_TOKENS, "num_ctx": 4096,
                 "temperature": 0.1, "keep_alive": "10m"},
    )
    return out["message"]["content"], sorted({m["source"] for m in metas})