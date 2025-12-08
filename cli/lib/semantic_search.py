from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def generate_embedding(self, text):
        text = text.strip()
        if not text or not text.strip(): raise ValueError("Text cannot be empty or contain only whitespace!")
        return self.model.encode([text])[0]

def verify_model():
    ss = SemanticSearch()
    print(f'Model loaded: {ss.model}')
    print(f'Max sequence length: {ss.model.max_seq_length}') 

def verify_command():
    return verify_model()


def embed_text(text):
    si = SemanticSearch()
    emb = si.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {emb[:3]}")
    print(f"Dimensions: {emb.shape[0]}")