import glob
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ye pdf ko padhega aur chunks me convert karega
def chunk_text(text, size=800, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

chunks = []
for path in glob.glob("data/*.pdf"):
    text = " ".join(page.extract_text() or "" for page in PdfReader(path).pages)
    chunks += chunk_text(text)

chunks = [c for c in chunks if len(c.strip()) > 100][:1000]
print("chunks:", len(chunks))

# ye chunks ko vectors me convert karega
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(chunks, normalize_embeddings=True)
vectors = np.array(vectors, dtype="float32")
print("vector shape:", vectors.shape)

# isme apan FAISS ka index bana rahe hain
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)
print("vectors in index:", index.ntotal)

# questions jo search karega
questions = [
    "What is the main topic of the document?",
    "How many questions are in the document?",
    "What is the document describing?",
    "What does it recommend?",
    "What numbers or statistics are given?",
]
for q in questions:
    qv = model.encode([q], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(qv, 3)
    print("\nQ:", q)
    for score, i in zip(scores[0], ids[0]):
        print(f"  {score:.2f} | {chunks[i][:120]!r}")

faiss.write_index(index, "rag.index")