from pathlib import Path
import json

def get_root(cwd = Path.cwd()):
    if cwd == cwd.parent: # reached root file system
        raise FileNotFoundError("No .git found")
    if (cwd/'.git').exists():
        return cwd
    return get_root(cwd.parent)


SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_CHUNK_SIZE = 200

PROJECT_ROOT = get_root()
DATA_PATH = PROJECT_ROOT / 'data/movies.json'
STOPWORDS_PATH = PROJECT_ROOT / 'data/stopwords.txt'
CACHE_DIR = PROJECT_ROOT / 'cache'
INDEX_PATH = CACHE_DIR / 'index.pkl'
DOCMAP_PATH = CACHE_DIR / 'docmap.pkl'
TERMFREQ_PATH = CACHE_DIR / 'term_frequencies.pkl'
DOC_LENGTHS_PATH = CACHE_DIR/ "doc_lengths.pkl"
MOVIE_EMBS_PATH = CACHE_DIR/"movie_embeddings.npy"



def load_movies() -> list[dict]:
    with open(get_root() / 'data/movies.json') as f:
        data = json.load(f)
    return data['movies']
    
def load_stopwords(path=STOPWORDS_PATH):
    with open(path) as f:
        return set(f.read().splitlines())