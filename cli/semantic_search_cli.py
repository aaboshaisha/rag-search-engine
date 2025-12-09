#!/usr/bin/env python3

import argparse
from lib.semantic_search import *

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands") 

    verify_parser = subparsers.add_parser('verify', help='Verify embedding model')
    
    embed_text_parser = subparsers.add_parser('embed_text', help='Embed text')
    embed_text_parser.add_argument('text', type=str, help='Text to be embedded')

    verify_embeddings_parser = subparsers.add_parser('verify_embeddings', help='Load or create document embeddings.')

    embedquery_parser = subparsers.add_parser('embedquery', help='Embed query text')
    embedquery_parser.add_argument('query', type=str, help='Query text to be embedded')

    search_parser = subparsers.add_parser('search', help='Sematic movie search')
    search_parser.add_argument('query', type=str, help='Search query string')
    search_parser.add_argument('--limit', type=int, nargs='?', help='Optional search limit paramater')

    chunk_parser = subparsers.add_parser('chunk', help='Split text into n sized chunks')
    chunk_parser.add_argument('text', type=str, help='Text to be chunked')
    chunk_parser.add_argument('--chunk-size', type=int, nargs='?', default = 200, help='Optional chunking parameter')
    
    args = parser.parse_args()
    
    match args.command:
        case "verify": return verify_command()
        case 'embed_text': return embed_text(args.text)
        case 'verify_embeddings': return verify_embeddings()
        case 'embedquery': return embedquery_command(args.query)
        case 'search': return search_command(args.query, args.limit)
        case 'chunk': return chunk_command(args.text, args.chunk_size)
        case _: parser.print_help()

if __name__ == "__main__":
    main()