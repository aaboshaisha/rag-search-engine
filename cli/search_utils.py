from pathlib import Path
import json


def get_root(cwd = Path.cwd()):
    if cwd == cwd.parent: # reached root file system
        raise FileNotFoundError("No .git found")
    if (cwd/'.git').exists():
        return cwd
    return get_root(cwd.parent)


PROJECT_ROOT = get_root()
DATA_PATH = PROJECT_ROOT / 'data/movies.json'
SEARCH_LIMIT = 5


def load_movies() -> list[dict]:
    with open(get_root() / 'data/movies.json') as f:
        data = json.load(f)
    return data['movies']
    
