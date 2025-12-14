import argparse
from hybrid_search import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser('normalize', help="Scores to normalize")
    normalize_parser.add_argument('scores', nargs='+')

    args = parser.parse_args()

    match args.command:
        case "normalize": normalize_command(args.scores)
        case _: parser.print_help()


if __name__ == "__main__":
    main()