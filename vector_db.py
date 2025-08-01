import faiss  # Ensure faiss is imported first
import numpy as np
import sqlite3
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import os

# Lifespan handler for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_index()
    yield
    save_index()

# Initialize FastAPI app
app = FastAPI(title="Vector Database API", lifespan=lifespan)

# Root endpoint to avoid 404
@app.get("/")
async def root():
    return {"message": "Welcome to the Vector Database API. Visit /docs for API documentation."}

# Initialize embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index parameters
dimension = 384  # MiniLM output dimension
index_file = "faiss_index.bin"
index = None
db_file = "metadata.db"

# Pydantic models
class VectorInput(BaseModel):
    text: str
    metadata: dict

class QueryInput(BaseModel):
    text: str
    top_k: int = 5
    metadata_filter: dict = None

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

# Initialize FAISS index with IndexIDMap
def init_index():
    global index
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
    else:
        index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
    return index

# Save FAISS index
def save_index():
    faiss.write_index(index, index_file)

# Add vector
@app.post("/add_vector")
async def add_vector(input: VectorInput):
    vector = embedder.encode([input.text])[0]
    vector = np.array([vector], dtype=np.float32)
    global index
    vector_id = index.ntotal
    index.add_with_ids(vector, np.array([vector_id], dtype=np.int64))
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("INSERT INTO vectors (id, text, metadata) VALUES (?, ?, ?)",
              (vector_id, input.text, str(input.metadata)))
    conn.commit()
    conn.close()
    save_index()
    return {"status": "success", "vector_id": vector_id}

# Query vectors
@app.post("/query")
async def query_vectors(input: QueryInput):
    query_vector = embedder.encode([input.text])[0]
    query_vector = np.array([query_vector], dtype=np.float32)
    distances, indices = index.search(query_vector, input.top_k)
    results = []
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    for idx, dist in zip(indices[0], distances[0]):
        query = "SELECT id, text, metadata FROM vectors WHERE id = ?"
        params = [int(idx)]
        if input.metadata_filter:
            for key, value in input.metadata_filter.items():
                query += f" AND metadata LIKE ?"
                params.append(f'%{key}:{value}%')
        c.execute(query, params)
        row = c.fetchone()
        if row:
            results.append({
                "id": row[0],
                "text": row[1],
                "metadata": eval(row[2]),
                "distance": float(dist)
            })
    conn.close()
    return {"results": results}

# Delete vector
@app.delete("/delete_vector/{vector_id}")
async def delete_vector(vector_id: int):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT id FROM vectors WHERE id = ?", (vector_id,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Vector not found")
    c.execute("DELETE FROM vectors WHERE id = ?", (vector_id,))
    conn.commit()
    conn.close()
    index.remove_ids(np.array([vector_id], dtype=np.int64))
    save_index()
    return {"status": "success"}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)