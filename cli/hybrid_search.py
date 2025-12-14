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
        
        keyword_results = normalize_list(keyword_results)
        semantic_results = normalize_list(semantic_results)
        
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
        keyword_results = self._bm25_search(query, 500 * limit)
        ss = self.semantic_search.search_chunks(query, 500 * limit)
        semantic_results = [(r['id'], r['score']) for r in ss] # get the results in the same format
        
        keyword_ids = [doc_id for doc_id, _ in keyword_results]
        semantic_ids = [doc_id for doc_id, _ in semantic_results]
        all_ids = set(keyword_ids) | set(semantic_ids)
        
        docmap_cache = {doc_id: {'document':self.semantic_search.docmap[doc_id]} for doc_id in all_ids} 
        
        combined_scores = dict()
        for doc_id in all_ids:
            document = docmap_cache[doc_id]
            keyword_rank, semantic_rank = get_rank(doc_id, keyword_ids), get_rank(doc_id, semantic_ids)
            rrf = combined_rrf(keyword_rank, semantic_rank, k)
            combined_scores[doc_id] = {'document': document['document'], 'keyword_rank': keyword_rank, 'semantic_rank': semantic_rank, 'rrf': rrf}
        
        sorted_results = sorted(combined_scores.values(), key=lambda x: x['rrf'], reverse=True)
        return sorted_results[:limit]


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


def rrf_score(rank, k=60):
    return 1 / (k + rank)

def get_rank(item, lst):
    return lst.index(item) if item in lst else None

def combined_rrf(keyword_rank, semantic_rank, k=60):
    rrf_k = rrf_score(keyword_rank, k) if keyword_rank else 0.0
    rrf_s = rrf_score(semantic_rank, k) if semantic_rank else 0.0
    return rrf_k + rrf_s

def format_results(results):
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['document']['title']}")
        print(f'RRF Score: {r["rrf"]:.4f}')
        print(f'BM25 Rank: {r["keyword_rank"]}, Semantic Rank: {r["semantic_rank"]}')
        print(r['document']['description'][:100])
        print('\n')

def rrf_search_command(query, k, limit):
    si = HybridSearch(load_movies())
    results = si.rrf_search(query, k, limit)
    format_results(results)