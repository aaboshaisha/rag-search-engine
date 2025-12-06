#!/usr/bin/env python3

import argparse
import json
from keyword_search import *



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index and save it to disk.")
    
    tf_parser = subparsers.add_parser("tf", help="Print term frequency for a term in document with given id.")
    tf_parser.add_argument("doc_id", type=int, help="Document Id")
    tf_parser.add_argument("term", type=str, help="Term")

    args = parser.parse_args()

    match args.command:
        case "search":
            matches = search(args.query)
            print(f'Searching for: {args.query}')
            for i, m in enumerate(matches):
                print(f'{i}. {m["title"]}')

        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")

        case "tf":
            return tf_command(args.doc_id, args.term)
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()