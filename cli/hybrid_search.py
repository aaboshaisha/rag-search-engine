from keyword_search import InvertedIndex
from semantic_search import ChunkedSemanticSearch
from search_utils import *

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not INDEX_PATH.exists():
            self.idx.build(); self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize(scores:list[float])->list[float]:
    min_score, max_score = min(scores), max(scores)
    if min_score == max_score:
        return [1.0 for _ in scores]
    f = lambda score: (score - min_score) / (max_score - min_score)
    return [f(score) for score in scores]

def normalize_command(scores=None):
    """accepts a list of scores and prints the normalized scores"""
    if not scores:
        return None
    scores = [float(s) for s in scores]
    norm_scores = normalize(scores)
    for score in norm_scores:
        print(f"* {score:.4f}")