import argparse
from hybrid_search import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser('normalize', help="Scores to normalize")
    normalize_parser.add_argument('scores', nargs='+')

    weighted_search_parser = subparsers.add_parser('weighted-search', help="Combined BM25 & Semantic searches")
    weighted_search_parser.add_argument('query', type=str, help='Query text')
    weighted_search_parser.add_argument('--alpha', type=float, nargs='?', default=0.5, help='Optional weight parameter. Defaults to 0.5')
    weighted_search_parser.add_argument('--limit', type=int, nargs='?', default=5, help='Optional results limit parameter. Defaults to 5')

    rrf_search_parser = subparsers.add_parser('rrf-search', help='Combined BM25 & Semantic using RRF')
    rrf_search_parser.add_argument('query', type=str, help='Query text')
    rrf_search_parser.add_argument('k', type=int, nargs='?', default=60, help='Optional parameter to control how much weight to give to higher vs lower rank results. Defaults to 60')
    rrf_search_parser.add_argument('--limit', type=int, nargs='?', default=5, help='Optional results limit parameter. Defaults to 5')
    

    args = parser.parse_args()

    match args.command:
        case "normalize": normalize_command(args.scores)
        case "weighted-search": weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search": rrf_search_command(args.query, args.k, args.limit)
        case _: parser.print_help()


if __name__ == "__main__":
    main()