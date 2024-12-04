import faiss

import numpy as np

def initialize_faiss(dimensions):
    index = faiss.IndexFlatL2(dimensions)
    return index

def add_to_faiss(index, embeddings):
    index.add(np.array(embeddings).astype("float32"))



def search_faiss(index, query_embedding, top_k=5):
    distances, indices = index.search(np.array([query_embedding]).astype("float32"), top_k)
    return distances, indices
