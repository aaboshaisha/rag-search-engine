from search_utils import *
import string, pickle, math
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
from pathlib import Path

stopwords = load_stopwords()
stemmer = PorterStemmer()

def tokenize(s:str) -> list[str]:
    """Tokenize. Remove stopwords. Returns word stems"""
    valid_toks = []
    for t in preprocess(s).split():
        if t not in stopwords:
            valid_toks.append(stemmer.stem(t))
    return valid_toks

def preprocess(s:str) -> str:
    """Lowercase and remove punctuation"""
    s = s.lower()
    return s.translate(str.maketrans("","",string.punctuation))

def has_matching_token(qry_toks:list[str], title_toks:list[str]) -> bool:
    """Checks if at least one token from the query matches any part of a token from the title"""
    for qt in qry_toks:
        for tt in title_toks:
            if qt in tt:
                return True
    return False

def search(qry:str, limit=SEARCH_LIMIT) -> list[str]:
    idx = InvertedIndex()
    idx.load()
    seen, docs = set(), []
    
    for tok in tokenize(qry):
        ids = idx.get_documents(tok)
        for i in ids:
            if i in seen:
                continue
            seen.add(i)
            docs.append(idx.docmap[i])
            if len(docs) >= limit:
                return docs
    return docs


class InvertedIndex:
    def __init__(self):
        self.index, self.docmap, self.term_frequencies, self.doc_lengths = defaultdict(set), dict(), defaultdict(Counter), dict()

    def __add_document(self, doc_id:int, text:str) -> None:
        """Tokenize the input text, then add each token to the index with the document ID."""
        toks = tokenize(text)
        self.doc_lengths[doc_id] = len(toks)
        self.term_frequencies[doc_id] = Counter(toks)
        for tok in toks:
            self.index[tok].add(doc_id)
            

    def __get_avg_doc_length(self) -> float:
        """Calculate and return the average document length across all documents"""
        if len(self.doc_lengths) == 0:
            return 0.0
        return sum(v for k,v in self.doc_lengths.items()) / len(self.doc_lengths) 
    
    def get_documents(self, term:str) -> list[int]:
        """get the set of document IDs for a given token, and return them as a list, sorted in ascending order."""
        return sorted(self.index[term])

    def get_tf(self, doc_id:int, term:str) -> int:
        """return the times the token appears in the document with the given ID."""
        counts = self.term_frequencies[doc_id]
        toks = tokenize(term)
        if len(toks) != 1:
            raise ValueError("Cannot search more than one term")
        return counts[toks[0]]

    def get_idf(self, term=str) -> int:
        total_doc_count = len(self.term_frequencies)
        tok = tokenize(term)[0]
        term_match_doc_count = sum(1 for doc_id in self.term_frequencies if tok in self.term_frequencies[doc_id])
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_tfidf(self, doc_id:int, term:str) -> float:
        tf, idf = self.get_tf(doc_id, term), self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term:str) -> float:
        toks = tokenize(term)
        if len(toks) != 1:
            raise ValueError("Input must be only one term")
        N = len(self.docmap)
        df = len(self.get_documents(toks[0]))
        IDF = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return IDF

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        
        # Length normalization factor
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        # Apply to term frequency
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        
        return tf_component

    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit):
        toks, bm25_scores = tokenize(query), dict()
        for doc_id in self.docmap.keys():
            bm25_scores[doc_id] = sum(self.bm25(doc_id, tok) for tok in toks)
        sorted_scores = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:limit]
    
    def build(self, movies=load_movies()):
        """iterate over all the movies and add them to both the index and the docmap."""
        for m in movies:
            doc_id, text = m['id'], f"{m['title']} {m['description']}"
            self.__add_document(doc_id, text) # add to index
            self.docmap[doc_id] = m
    
    def save(self):
        def save_to_pickle(obj, path):
            fpath = Path(path)
            fpath.parent.mkdir(parents=True, exist_ok=True)
            with open(fpath, 'wb') as f:
                pickle.dump(obj, f)
                print(f'Saved at {path}')
        
        save_to_pickle(self.index, INDEX_PATH)
        save_to_pickle(self.docmap, DOCMAP_PATH)
        save_to_pickle(self.term_frequencies, TERMFREQ_PATH)
        save_to_pickle(self.doc_lengths, DOC_LENGTHS_PATH)
    
    def load(self):
        with open(INDEX_PATH, 'rb') as f: self.index = pickle.load(f)        
        with open(DOCMAP_PATH, 'rb') as f: self.docmap = pickle.load(f)
        with open(TERMFREQ_PATH, 'rb') as f: self.term_frequencies = pickle.load(f)
        with open(DOC_LENGTHS_PATH, 'rb') as f: self.doc_lengths = pickle.load(f)

def build_command()->None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents('merida')
    print(f"First document for token 'merida' = {docs[0]}") 

def tf_command(doc_id:int, term:str)-> None:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)


def idf_command(term:str) -> float:
    """Return IDF for given term"""
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)


def tfidf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    return idx.get_tfidf(doc_id, term)

def bm25idf_command(term:str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25tf_command(doc_id, term, b=None):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term)

def bm25search_command(query, limit=5):
    idx = InvertedIndex()
    idx.load()
    matches = idx.bm25_search(query, limit)
    matches = ((doc_id, score, idx.docmap[doc_id]['title']) for doc_id, score in matches)
    return matches
