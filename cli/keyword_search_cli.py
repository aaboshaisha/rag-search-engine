#!/usr/bin/env python3

import argparse
import json
from keyword_search import *



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            matches = search(args.query)
            print(f'Searching for: {args.query}')
            for i, m in enumerate(matches):
                print(f'{i}. {m["title"]}')
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()