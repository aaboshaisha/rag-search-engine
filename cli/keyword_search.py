from search_utils import *
import string, pickle
from nltk.stem import PorterStemmer
from collections import defaultdict
from pathlib import Path

stopwords = load_stopwords()
stemmer = PorterStemmer()

def tokenize(s:str) -> list[str]:
    """Tokenize. Remove stopwords. Returns word stems"""
    valid_toks = set()
    for t in preprocess(s).split():
        if t not in stopwords:
            valid_toks.add(stemmer.stem(t))
    return valid_toks

def preprocess(s:str) -> str:
    """Lowercase and remove punctuation"""
    s = s.lower()
    return s.translate(str.maketrans("","",string.punctuation))

# def has_matching_token(qry:str, title:str) -> bool:
#     """Checks if at least one token from the query matches any part of title"""
#     toks = tokenize(qry)
#     return any(tok in title for tok in toks)

def has_matching_token(qry_toks:list[str], title_toks:list[str]) -> bool:
    """Checks if at least one token from the query matches any part of a token from the title"""
    for qt in qry_toks:
        for tt in title_toks:
            if qt in tt:
                return True
    return False


def search(qry:str, limit=SEARCH_LIMIT) -> list[str]:
    movies = load_movies()
    matches = []
    for m in movies:
        qry_toks, title_toks = tokenize(qry), tokenize(m['title'])
        if has_matching_token(qry_toks, title_toks):
            print(m['title'])
            matches.append(m)
            if len(matches) >= limit:
                break
    return matches


class InvertedIndex:
    def __init__(self):
        self.index, self.docmap = defaultdict(set), dict()

    def __add_document(self, doc_id:int, text:str) -> None:
        """Tokenize the input text, then add each token to the index with the document ID."""
        toks = tokenize(text)
        for tok in toks:
            self.index[tok].add(doc_id)

    def get_documents(self, term:str) -> list[int]:
        """get the set of document IDs for a given token, and return them as a list, sorted in ascending order."""
        return sorted(self.index[term])

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
        save_to_pickle(self.index, CACHE_DIR/'index.pkl')
        save_to_pickle(self.docmap, CACHE_DIR/'docmap.pkl')


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents('merida')
    print(f"First document for token 'merida' = {docs[0]}") 