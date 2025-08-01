import faiss
import numpy as np
import sqlite3
import json
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
        init_index()
        yield
        save_index()
    except Exception as e:
        logger.error(f"Error in lifespan: {e}")
        raise

app = FastAPI(title="Vector Database API", lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Welcome to the Vector Database API. Visit /docs for API documentation."}

embedder = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index_file = "faiss_index.bin"
index = None
db_file = "metadata.db"

class VectorInput(BaseModel):
    text: str
    metadata: dict

class QueryInput(BaseModel):
    text: str
    top_k: int = 5
    metadata_filter: dict = None

def init_db():
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS vectors
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      text TEXT NOT NULL,
                      metadata TEXT)''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def init_index():
    global index
    try:
        if os.path.exists(index_file):
            os.remove(index_file)
            logger.info("Deleted existing faiss_index.bin for clean start")
        base_index = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap(base_index)
        logger.info(f"Created new FAISS IndexIDMap with base index {type(base_index).__name__}")
        return index
    except Exception as e:
        logger.error(f"Error initializing FAISS index: {e}")
        raise

def save_index():
    try:
        faiss.write_index(index, index_file)
        logger.info("FAISS index saved successfully")
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}")
        raise

@app.post("/add_vector")
async def add_vector(input: VectorInput):
    try:
        vector = embedder.encode([input.text])[0]
        vector = np.array([vector], dtype=np.float32)
        global index
        # Insert into SQLite first to get auto-generated ID
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute("INSERT INTO vectors (text, metadata) VALUES (?, ?)",
                  (input.text, json.dumps(input.metadata)))
        vector_id = c.lastrowid
        conn.commit()
        conn.close()
        # Add to FAISS with SQLite ID
        logger.info(f"Adding vector with id {vector_id}, shape {vector.shape}")
        index.add_with_ids(vector, np.array([vector_id], dtype=np.int64))
        save_index()
        logger.info(f"Added vector with id {vector_id}")
        return {"status": "success", "vector_id": vector_id}
    except Exception as e:
        logger.error(f"Error in add_vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_vectors(input: QueryInput):
    try:
        query_vector = embedder.encode([input.text])[0]
        query_vector = np.array([query_vector], dtype=np.float32)
        distances, indices = index.search(query_vector, input.top_k)
        results = []
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        logger.info(f"FAISS search returned indices: {indices[0].tolist()}, distances: {distances[0].tolist()}")
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                logger.warning(f"Invalid index {idx} returned by FAISS")
                continue
            c.execute("SELECT id, text, metadata FROM vectors WHERE id = ?", (int(idx),))
            row = c.fetchone()
            if row:
                try:
                    metadata = json.loads(row[2])
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error for id {idx}: {e}")
                    continue
                if input.metadata_filter:
                    # Relaxed comparison: convert both to strings
                    if all(str(metadata.get(key)) == str(value) for key, value in input.metadata_filter.items()):
                        results.append({
                            "id": row[0],
                            "text": row[1],
                            "metadata": metadata,
                            "distance": float(dist)
                        })
                        logger.info(f"Matched vector id {row[0]} with metadata {metadata}")
                else:
                    results.append({
                        "id": row[0],
                        "text": row[1],
                        "metadata": metadata,
                        "distance": float(dist)
                    })
                    logger.info(f"Matched vector id {row[0]} with metadata {metadata}")
        conn.close()
        logger.info(f"Query returned {len(results)} results")
        return {"results": results}
    except Exception as e:
        logger.error(f"Error in query_vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_vector/{vector_id}")
async def delete_vector(vector_id: int):
    try:
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
        logger.info(f"Deleted vector with id {vector_id}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error in delete_vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug_index")
async def debug_index():
    return {"total_vectors": index.ntotal}

@app.get("/debug_db")
async def debug_db():
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute("SELECT id, text, metadata FROM vectors")
        rows = c.fetchall()
        results = []
        for row in rows:
            try:
                metadata = json.loads(row[2])
            except json.JSONDecodeError:
                metadata = {"error": "Invalid JSON"}
            results.append({"id": row[0], "text": row[1], "metadata": metadata})
        conn.close()
        return {"vectors": results}
    except Exception as e:
        logger.error(f"Error in debug_db: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
