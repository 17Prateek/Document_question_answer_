from sentence_transformers import SentenceTransformer

def generate_embeddings(text, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    sentences = text.split("\n")

    embeddings = model.encode(sentences)
    
    return sentences, embeddings
