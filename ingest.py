import glob, hashlib
from pathlib import Path
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
DB_DIR = SCRIPT_DIR / "vectordb"
CHUNK, OVERLAP = 800, 150
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_text(path):
    if path.lower().endswith(".pdf"):
        parts = []
        with pdfplumber.open(path) as pdf:
            for pg in pdf.pages:
                parts.append(pg.extract_text() or "")
                for tbl in pg.extract_tables():
                    parts.append("\n".join(" | ".join(c or "" for c in row) for row in tbl))
        return "\n".join(parts)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk(text):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+CHUNK])
        i += CHUNK - OVERLAP
    return [c.strip() for c in out if c.strip()]

def main():
    client = chromadb.PersistentClient(str(DB_DIR))
    col = client.get_or_create_collection("fab_docs")
    for path in glob.glob(str(DATA_DIR / "**" / "*"), recursive=True):
        if not Path(path).is_file():
            continue
        chunks = chunk(load_text(path))
        if not chunks:
            continue
        name = Path(path).name
        ids = [hashlib.md5(f"{name}{j}".encode()).hexdigest() for j in range(len(chunks))]
        embs = embedder.encode(chunks, show_progress_bar=False).tolist()
        col.upsert(ids=ids, documents=chunks, embeddings=embs,
                   metadatas=[{"source": name} for _ in chunks])
        print(f"{name}: {len(chunks)} chunks")

if __name__ == "__main__":
    main()
