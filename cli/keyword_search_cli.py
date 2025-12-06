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
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser("idf", help="Calculate IDF for a given term")
    idf_parser.add_argument("term", type=str, help="Term to calculate IDF for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate TF-IDF score for a given document ID and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="term to calculate TF-IDF for")

    bm25_idf_parser = subparsers.add_parser('bm25idf', help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

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
            tf = tf_command(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {tf}")

        case "idf":
            print(f'{idf_command(args.term)}:.2f')

        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            bm25idf = bm25idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()