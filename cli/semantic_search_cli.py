#!/usr/bin/env python3

import argparse
from lib.semantic_search import *

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands") 

    verify_parser = subparsers.add_parser('verify', help='Verify embedding model')
    
    embed_text_parser = subparsers.add_parser('embed_text', help='Embed text')
    embed_text_parser.add_argument('text', type=str, help='Text to be embedded')
    

    args = parser.parse_args()
    
    match args.command:
        case "verify": return verify_command()
        case 'embed_text': return embed_text(args.text)
        case _: parser.print_help()

if __name__ == "__main__":
    main()