from search_utils import load_movies, SEARCH_LIMIT

def search(qry:str, limit=SEARCH_LIMIT) -> list[str]:
    movies = load_movies()
    matches = []
    for m in movies:
        if qry in m['title']:
            matches.append(m)
            if len(matches) >= limit:
                break
    return matches