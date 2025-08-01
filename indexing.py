import faiss
import numpy as np
import os
import sqlite3

dimension = 128  # Set this to the dimensionality of your vectors
nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)


index_file = "faiss.index"  # Define the path to your index file
db_file = "vectors.db"  # Define the path to your SQLite database file

def init_index():
    global index
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
    else:
        nlist = 100
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute("SELECT text FROM vectors LIMIT 10000")
        texts = [row[0] for row in c.fetchall()]
        if texts:
            # Define or import your embedder here. Example using sentence-transformers:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            sample_vectors = embedder.encode(texts)
            index.train(np.array(sample_vectors, dtype=np.float32))
        conn.close()
    return index
index.train(np.random.rand(10000, dimension).astype(np.float32))  # Train on sample data