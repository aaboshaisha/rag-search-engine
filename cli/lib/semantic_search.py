from sentence_transformers import SentenceTransformer
import numpy as np
from search_utils import *

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings, self.documents, self.document_map = None, None, dict()
    
    def generate_embedding(self, text):
        text = text.strip()
        if not text or not text.strip(): raise ValueError("Text cannot be empty or contain only whitespace!")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents
        strs = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            strs.append(f"{doc['title']}: {doc['description']}")
        
        self.embeddings = self.model.encode(strs, show_progress_bar=True)
        assert self.embeddings.shape == (len(documents), 384)
        np.save(MOVIE_EMBS_PATH, self.embeddings)
        print(f'Successfuly saved embeddings at {MOVIE_EMBS_PATH}')
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc
        
        if MOVIE_EMBS_PATH.exists():
            self.embeddings = np.load(MOVIE_EMBS_PATH)
            if self.embeddings.shape[0] == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

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

def verify_embeddings():
    documents = load_movies()
    si = SemanticSearch()
    embeddings = si.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query_text(query):
    si = SemanticSearch()
    embedding = si.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
    return embedding

def embedquery_command(query:str):
    return embed_query_text(query)