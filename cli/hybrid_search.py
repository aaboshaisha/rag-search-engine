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
        keyword_results = self._bm25_search(query, 500 * limit)
        ss = self.semantic_search.search_chunks(query, 500 * limit)
        semantic_results = [(r['id'], r['score']) for r in ss] # get the results in the same format
        
        keyword_results = normalize_results(keyword_results)
        semantic_results = normalize_results(semantic_results)
        
        common_ids = set(doc_id for doc_id, _ in keyword_results) & set(doc_id for doc_id, _ in semantic_results)
        docmap_cache = {doc_id: self.semantic_search.docmap[doc_id] for doc_id in common_ids}
        
        combined_scores = {
            doc_id: {'document': docmap_cache[doc_id],
                    'keyword_score':dict(keyword_results).get(doc_id, float('-inf')),
                    'semantic_score':dict(semantic_results).get(doc_id, float('-inf'))}
            for doc_id in common_ids
        }
        
        for entry in combined_scores.values():
            entry['hybrid_score'] = hybrid_score(entry['keyword_score'], entry['semantic_score'])
        
        
        sorted_results = sorted(combined_scores.values(), key=lambda x: x['hybrid_score'], reverse=True)
        return sorted_results

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

def normalize_results(results:list[tuple]) -> list[tuple]:
    ids, scores = zip(*results)
    return list(zip(ids, normalize(scores)))

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def weighted_search_command(query, alpha=0.5, limit=5):
    si = HybridSearch(load_movies()) 
    results = si.weighted_search(query, alpha, limit)
    for i, r in enumerate(results[:limit]):
        print(f"{i+1}. {r['document']['title']}")
        print(f'Hybrid Score: {r["hybrid_score"]:.4f}')
        print(f'BM25: {r["keyword_score"]:.4f}, Semantic: {r["semantic_score"]:.4f}')
        print(r['document']['description'][:100])