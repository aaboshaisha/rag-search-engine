from search_utils import *
import string, pickle
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
        self.index, self.docmap, self.term_frequencies = defaultdict(set), dict(), defaultdict(Counter)

    def __add_document(self, doc_id:int, text:str) -> None:
        """Tokenize the input text, then add each token to the index with the document ID."""
        toks = tokenize(text)
        for tok in toks:
            self.index[tok].add(doc_id)
            self.term_frequencies[doc_id][tok] += 1 # update term freq for each token

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
    
    def load(self):
        with open(INDEX_PATH, 'rb') as f:
            self.index = pickle.load(f)
            print('Index loaded')
        
        with open(DOCMAP_PATH, 'rb') as f:
            self.docmap = pickle.load(f)
            print('Docmap loaded')
    
        with open(TERMFREQ_PATH, 'rb') as f:
            self.term_frequencies = pickle.load(f)
            print('Term frequencies loaded')        


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents('merida')
    print(f"First document for token 'merida' = {docs[0]}") 

def tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    print(idx.get_tf(doc_id, term))