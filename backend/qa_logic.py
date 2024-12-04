from backend.database import search_faiss
from sentence_transformers import SentenceTransformer

def find_relevant_chunks(query, index, sentences, model_name="all-MiniLM-L6-v2"):
    
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])[0]
    
    distances, indices = search_faiss(index, query_embedding, top_k=5)
    results = [sentences[i] for i in indices[0]]
    return results
