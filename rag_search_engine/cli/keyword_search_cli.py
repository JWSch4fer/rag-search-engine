#!/usr/bin/env python3

import argparse

from rag_search_engine.utils.search import (
    basic_search,
    calc_bm25_freq,
    calc_freq,
    calc_idf,
    search_inverted_index,
    calc_tf_idf,
    calc_bm25,
)
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
    invidx = InvertedIndex()
    if invidx.exists():
        invidx.load()
        docmap = invidx.docmap()
        postings = invidx.index()
        print("Using cached data...")
        result = search_inverted_index(args.query, postings)
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
        print(
            "{:} appears {:}".format(
                args.word, calc_freq(args.word, args.doc_id, docmap)
            )
        )
    else:
        raise Exception("Have to build a cached database first")


def handle_bm25_frequency(args):
    invidx = InvertedIndex()
    if invidx.exists():
        invidx.load()
        docmap = invidx.docmap()
        print(
            f"BM25 TF score of '{args.word}' in document '{args.doc_id}': {calc_bm25_freq(args.word, args.doc_id, docmap, args.k1):.2f}"
        )
    else:
        raise Exception("Have to build a cached database first")


def handle_idf(args):
    invidx = InvertedIndex()
    if invidx.exists():
        invidx.load()
        docmap = invidx.docmap()
        postings = invidx.index()
        idf = calc_idf(args.word, postings, docmap)
        print(f"Inverse document frequency of {args.word}: {idf:.2f}")
    else:
        raise Exception("Have to build a cached database first")


def handle_bm25(args):
    invidx = InvertedIndex()
    if invidx.exists():
        invidx.load()
        docmap = invidx.docmap()
        postings = invidx.index()
        idf = calc_bm25(args.word, postings, docmap)
        print(f"BM25 IDF score of {args.word}: {idf:.2f}")
    else:
        raise Exception("Have to build a cached database first")


def handle_tfidf(args):
    invidx = InvertedIndex()
    if invidx.exists():
        invidx.load()
        docmap = invidx.docmap()
        postings = invidx.index()
        tf_idf = calc_tf_idf(args.doc_id, args.word, postings, docmap)
        print(
            f"TF-IDF score of '{args.word}' in document '{args.doc_id}': {tf_idf:.2f}"
        )
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
    # ____________________search command______________________________________________
    # ________________________________________________________________________________
    search_p = subparsers.add_parser("search", help="Search movies using BM25")
    search_p.add_argument("query", type=str, help="Search query")
    # attach handler
    search_p.set_defaults(func=handle_search)
    # ________________________________________________________________________________
    # ____________________frequency of a word_________________________________________
    # ________________________________________________________________________________
    tf_p = subparsers.add_parser("tf", help="get the frequency of a word")
    tf_p.add_argument("doc_id", type=int, help="doc id to search")
    tf_p.add_argument("word", type=str, help="word to get frequency")
    # attach handler
    tf_p.set_defaults(func=handle_frequency)
    # ________________________________________________________________________________
    # ______________inverse document frequency of a word______________________________
    # ________________________________________________________________________________
    idf_p = subparsers.add_parser("idf", help="get the inverse document frequency")
    idf_p.add_argument("word", type=str, help="word to get frequency")
    # attach handler
    idf_p.set_defaults(func=handle_idf)
    # ________________________________________________________________________________
    # ____________________tf-idf of a word_________________________________________
    # ________________________________________________________________________________
    tfidf_p = subparsers.add_parser("tfidf", help="get the tf-idf score")
    tfidf_p.add_argument("doc_id", type=int, help="doc id to search")
    tfidf_p.add_argument("word", type=str, help="word to get frequency")
    # attach handler
    tfidf_p.set_defaults(func=handle_tfidf)
    # ________________________________________________________________________________
    # ____________________BM25 of a word_________________________________________
    # ________________________________________________________________________________
    bm25_p = subparsers.add_parser("bm25idf", help="get the bm25idf score")
    bm25_p.add_argument("word", type=str, help="word to get frequency")
    # attach handler
    bm25_p.set_defaults(func=handle_bm25)
    # ________________________________________________________________________________
    # ____________________bm25 frequency of a word____________________________________
    # ________________________________________________________________________________
    bm25tf_p = subparsers.add_parser("bm25tf", help="get the bm25 frequency of a word")
    bm25tf_p.add_argument("doc_id", type=int, help="doc id to search")
    bm25tf_p.add_argument("word", type=str, help="word to get frequency")
    bm25tf_p.add_argument(
        "-k1", type=float, default=1.5, help="scaling parameter for frequency: default=1.5"
    )
    # attach handler
    bm25tf_p.set_defaults(func=handle_bm25_frequency)

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
