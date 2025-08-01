nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index.train(np.random.rand(10000, dimension).astype(np.float32))  # Train on sample data