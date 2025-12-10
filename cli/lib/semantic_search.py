from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
from search_utils import *
import re

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
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
        self.embeddings = self.build_embeddings(documents)
        return self.embeddings

    def search(self, query, limit):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents loaded. Call `load_or_create_embeddings` first.")


        query_emb = self.generate_embedding(query)
        dot_prods = query_emb @ self.embeddings.T
    
        norms_prods = norm(query_emb) * norm(self.embeddings, axis=1) # (5000,) vector
        sim_scores = dot_prods / norms_prods
    
        docs_and_scores = [(doc, score) for doc, score in zip(self.documents, sim_scores)]
        docs_and_scores.sort(key=lambda x: x[1], reverse=True)
    
        results = [{'score':tup[1], 'title':tup[0]['title'], 'description':tup[0]['description']} for tup in docs_and_scores[:limit]]
        return results

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

def search_command(query:str, limit:int=5):
    si = SemanticSearch()
    _ = si.load_or_create_embeddings(load_movies())
    results = si.search(query, limit)
    
    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()

def fixed_chunk(text:str, chunk_size:int=DEFAULT_CHUNK_SIZE, overlap:int=0):
    assert overlap < chunk_size, f'Overlap {overlap} must be < Chunk size {chunk_size}'
    words = text.split()
    i, chunks = 0, []
    while i < len(words) - overlap:
        chunks.append(words[i:i+chunk_size])
        i += chunk_size - overlap
    return chunks

def chunk_command(text:str, chunk_size:int|None, overlap:int|None):
    print(f'Chunking {len(text)} characters')
    chunks = fixed_chunk(text, chunk_size, overlap)
    for i, chunk in enumerate(chunks):
        print(f'{i+1}.', ' '.join(chunk))


def semantic_chunk(text:str, max_chunk_size:int=4, overlap:int=0):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    assert overlap < max_chunk_size, f'Overlap {overlap} must be < Chunk size {max_chunk_size}'
    i, chunks = 0, []
    while i < len(sentences) - overlap:
        chunks.append(' '.join(sentences[i:i+max_chunk_size]))
        i += max_chunk_size - overlap
    return chunks

def semantic_chunk_command(text:str, max_chunk_size:int|None, overlap:int|None):
    print(f'Semantically chunking {len(text)} characters')
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    for i, chunk in enumerate(chunks):
        print(f'{i+1}.', chunk)


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents:list[dict])->np.ndarray:
        self.documents = documents
        self.docmap = {doc['id']: doc for doc in documents} 
        all_chunks, metadata = [], []
        for i, doc in enumerate(documents):
            text = doc['description']
            if not text or not text.strip():
                continue
            chunks = semantic_chunk(text, 4, 1)
            for j, chunk in enumerate(chunks):
                meta = {'movie_idx': i, 'chunk_idx': j, 'total_chunks': len(chunks)}
                all_chunks.append(chunk); metadata.append(meta)

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = metadata
        assert len(all_chunks) == len(self.chunk_metadata) == self.chunk_embeddings.shape[0], \
        f"Mismatch in chunk data: {len(all_chunks)} chunks, {len(self.chunk_metadata)} metadata entries, {self.chunk_embeddings.shape[0]} embeddings"

        np.save(CHUNK_EMBS_PATH, self.chunk_embeddings)
        with open(CHUNK_META_PATH, 'w') as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks), "documents": len(self.documents)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.docmap = {doc['id']: doc for doc in documents} 
        if CHUNK_EMBS_PATH.exists() and CHUNK_META_PATH.exists():
            with open(CHUNK_META_PATH, 'r') as f:
                data = json.load(f)
                self.chunk_metadata, total_chunks, total_documents = data['chunks'], data['total_chunks'], data['documents']
                if len(documents) == total_documents:
                    self.chunk_embeddings = np.load(CHUNK_EMBS_PATH)
                    return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

def embed_chunks_command():
    movies = load_movies()
    csi = ChunkedSemanticSearch()
    embeddings = csi.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")
