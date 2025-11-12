#!/usr/bin/env python3

import argparse, math

from rag_search_engine.utils.search import basic_search, calc_idf, search_inverted_index
from rag_search_engine.utils.inv_idx import InvertedIndex


def handle_build(args) -> None:
    # Put your build logic here
    # e.g., idx = InvertedIndex(...); idx.build(...); idx.save(...)
    print("Building the inverted index...")
    print("force:", args.force, "data:", args.data)
    invidx = InvertedIndex(args.data)
    if not invidx.exists() or args.force:
        invidx.build()
        invidx.save()

    # # TODO: check if pkl files match json source
    # TODO: add option for a partial build?


def handle_search(args) -> None:
    print("Searching for:", args.query)
    # update so InvertedIndex can be initialized without providing a file path
    # implement search based on inverted index
    invidx = InvertedIndex()
    if invidx.exists():
        invidx.load()
        docmap = invidx.docmap()
        postings = invidx.index()
        print("Using cached data...")
        result = search_inverted_index(args.query, postings, docmap)
        for doc_idx in result:
            print("{:}. {:}".format(doc_idx, docmap[doc_idx]["title"]))
        return

    print("No cached data, falling back to basic search")
    results = basic_search(args.query.lower())
    for idx, r in enumerate(results):
        print(f"{idx}. {r}")

def handle_frequency(args):
    invidx = InvertedIndex()
    if invidx.exists():
        invidx.load()
        docmap = invidx.docmap()
        print("{:} appears {:}".format(args.word,docmap[args.doc_id]["description"].count(args.word)))
    else:
        raise Exception("Have to build a cached database first")

def handle_idf(args):
    invidx = InvertedIndex()
    if invidx.exists():
        invidx.load()
        docmap = invidx.docmap()
        postings = invidx.index()
        idf = calc_idf(args.word, postings,docmap)
    else:
        raise Exception("Have to build a cached database first")



def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ________________________________________________________________________________
    # ____________________build command_______________________________________________
    # ________________________________________________________________________________
    build_p = subparsers.add_parser("build", help="Build the inverted index")
    build_p.add_argument(
        "data",
        type=str,
        help="Path to source data (default: %(default)s)",
    )
    build_p.add_argument(
        "-f",
        "--force",
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

    # ________________________________________________________________________________
    # ____________________frequency of a word_________________________________________
    # ________________________________________________________________________________
    search_p = subparsers.add_parser("tf", help="get the frequency of a word")
    search_p.add_argument("doc_id", type=int, help="doc id to search")
    search_p.add_argument("word", type=str, help="word to get frequency")
    # attach handler
    search_p.set_defaults(func=handle_frequency)
    # ________________________________________________________________________________

    # ________________________________________________________________________________
    # ____________________frequency of a word_________________________________________
    # ________________________________________________________________________________
    search_p = subparsers.add_parser("idf", help="get the inverse document frequency")
    search_p.add_argument("word", type=str, help="word to get frequency")
    # attach handler
    search_p.set_defaults(func=handle_idf)
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
