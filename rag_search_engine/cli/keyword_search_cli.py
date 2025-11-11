#!/usr/bin/env python3

import argparse, json

from rag_search_engine.utils.search import basic_search
from rag_search_engine.utils.inv_idx import InvertedIndex

def handle_build(args):
    # Put your build logic here
    # e.g., idx = InvertedIndex(...); idx.build(...); idx.save(...)
    print("Building the inverted index...")
    print("force:", args.force, "data:", args.data)
    invidx = InvertedIndex(args.data)
    if not invidx.load() or args.force:
        invidx.build()
        invidx.save()



def handle_search(args):
    print("Searching for:", args.query)
    results = basic_search(args.query.lower())
    for idx, r in enumerate(results):
        print(f"{idx}. {r}")



def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ________________________________________________________________________________
    # ____________________build command_______________________________________________
    # ________________________________________________________________________________
    build_p = subparsers.add_parser("build", help="Build the inverted index")
    build_p.add_argument(
        "-d","--data",
        type=str,
        default="data/movies.json",
        help="Path to source data (default: %(default)s)",
    )
    build_p.add_argument(
        "-f", "--force",
        action="store_true",
        help="Rebuild cache even if a cached index is present",
    )
    # attach handler
    build_p.set_defaults(func=handle_build)
    # ________________________________________________________________________________

    # ________________________________________________________________________________
    # ____________________search command______________________________________________
    # ________________________________________________________________________________
    search_p = subparsers.add_parser("search", help="Search movies using BM25")
    search_p.add_argument("query", type=str, help="Search query")
    # attach handler
    search_p.set_defaults(func=handle_search)
    # ________________________________________________________________________________
    args = parser.parse_args()
    
    return parser

def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    # print(sys.argv, args)  # <-- uncomment to debug
    args.func(args)

if __name__ == "__main__":
    main()
