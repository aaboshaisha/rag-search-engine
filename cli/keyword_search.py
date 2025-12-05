from search_utils import load_movies, SEARCH_LIMIT
import string

def remove_punc(s:str) -> str:
    return s.translate(str.maketrans("","",string.punctuation))

def tokenize(s:str) -> list[str]:
    return [t for t in preprocess(s).split()]
    
def preprocess(s:str) -> str:
    return remove_punc(s.lower())

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