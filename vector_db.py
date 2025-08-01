import faiss
import numpy as np
import sqlite3
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import os

# Initialize FastAPI app
app = FastAPI(title="Vector Database API")

# Initialize embedding model (MiniLM for lightweight embeddings)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index parameters
dimension = 384  # MiniLM output dimension
index_file = "faiss_index.bin"
index = None
db_file = "metadata.db"

# Pydantic models for API requests
class VectorInput(BaseModel):
    text: str
    metadata: dict

class QueryInput(BaseModel):
    text: str
    top_k: int = 5

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS vectors
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  text TEXT NOT NULL,
                  metadata TEXT)''')
    conn.commit()
    conn.close()

# Load or create FAISS index
def init_index():
    global index
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
    else:
        index = faiss.IndexFlatL2(dimension)  # L2 distance for simplicity
    return index

# Save FAISS index to disk
def save_index():
    faiss.write_index(index, index_file)

# Add vector to FAISS and metadata to SQLite
@app.post("/add_vector")
async def add_vector(input: VectorInput):
    # Embed text
    vector = embedder.encode([input.text])[0]
    vector = np.array([vector], dtype=np.float32)

    # Add to FAISS
    global index
    vector_id = index.ntotal
    index.add(vector)

    # Save metadata to SQLite
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("INSERT INTO vectors (id, text, metadata) VALUES (?, ?, ?)",
              (vector_id, input.text, str(input.metadata)))
    conn.commit()
    conn.close()

    # Save index
    save_index()
    return {"status": "success", "vector_id": vector_id}

# Query FAISS index
@app.post("/query")
async def query_vectors(input: QueryInput):
    # Embed query
    query_vector = embedder.encode([input.text])[0]
    query_vector = np.array([query_vector], dtype=np.float32)

    # Search FAISS
    distances, indices = index.search(query_vector, input.top_k)

    # Fetch metadata from SQLite
    results = []
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    for idx, dist in zip(indices[0], distances[0]):
        c.execute("SELECT id, text, metadata FROM vectors WHERE id = ?", (int(idx),))
        row = c.fetchone()
        if row:
            results.append({
                "id": row[0],
                "text": row[1],
                "metadata": eval(row[2]),  # Convert string back to dict
                "distance": float(dist)
            })
    conn.close()
    return {"results": results}

# Delete vector
@app.delete("/delete_vector/{vector_id}")
async def delete_vector(vector_id: int):
    # Check if vector exists
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT id FROM vectors WHERE id = ?", (vector_id,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Vector not found")

    # Delete from SQLite
    c.execute("DELETE FROM vectors WHERE id = ?", (vector_id,))
    conn.commit()
    conn.close()

    # Rebuild FAISS index (simplified; for production, use incremental updates)
    global index
    index = faiss.IndexFlatL2(dimension)
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT id, text FROM vectors")
    rows = c.fetchall()
    vectors = []
    for row in rows:
        vector = embedder.encode([row[1]])[0]
        vectors.append(vector)
    if vectors:
        index.add(np.array(vectors, dtype=np.float32))
    conn.close()
    save_index()
    return {"status": "success"}

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    init_index()

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)