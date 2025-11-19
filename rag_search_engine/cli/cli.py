#!/usr/bin/env python3
import argparse

from rag_search_engine.utils.semantic_search import SemanticSearch
from rag_search_engine.utils.keyword_search import KeywordSearch
import os
from dotenv import load_dotenv

from rag_search_engine.utils.basesearch_db import DEFAULT_DB_PATH

# load_dotenv()
# api_key = os.environ.get("GEMINI_API_KEY")


def handle_build(args: argparse.Namespace) -> None:
    # Build DB for keyword search
    ks = KeywordSearch.build_from_docs(
        docs_path=args.file_path, db_path=DEFAULT_DB_PATH, force=args.force
    )
    # print(f"SQLite DB path: {ks.db_path}")
    # print(f"Docs source:    {ks.docs_path}")
    # print(f"Number of docs: {len(ks.documents)}")
    # print(f"Movies table:   {ks.count_movies()} rows")
    ks.verify_db()
    ks.close()
    print("--------------------------------------------------------------")
    # Build DB for semantic search
    ssc = SemanticSearch.build_from_docs(
        docs_path=args.file_path, db_path=DEFAULT_DB_PATH, force=args.force
    )
    # print(f"SemanticSearch dimensions: {ssc.embedding_dim}")
    ssc.verify_db()
    ssc.close()


def handle_keyword_search(args: argparse.Namespace) -> None:
    ks = KeywordSearch(
        docs_path=None,  # open existing DB no path
        db_path=DEFAULT_DB_PATH,  # same path you built to
    )
    results = ks.search(args.query, k=args.limit)
    for rank, meta in enumerate(results, start=1):
        print(f"{rank:2d}. {meta['score']:.4f}  {meta['title']}")
    ks.close()


def handle_semantic_search(args: argparse.Namespace) -> None:
    ssc = SemanticSearch(
        docs_path=None,  # open existing DB no path
        db_path=DEFAULT_DB_PATH,  # same path you built to
    )
    results = ssc.query_top_k(args.query, k=args.limit)
    for rank, meta in enumerate(results, start=1):
        print(f"{rank:2d}. {meta['distance']:.4f}  {meta['title']}")
    ssc.close()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    # ________________________________________________________________________________
    # ___________________________global utilities_____________________________________
    # ________________________________________________________________________________
    build_p = subparsers.add_parser("build", help="Build the inverted index")
    build_p.add_argument(
        "file_path",
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
    # ___________________________keyword search_____________________________________
    # ________________________________________________________________________________
    build_ks = subparsers.add_parser("key_search", help="bm25 based search")
    build_ks.add_argument(
        "query",
        type=str,
        help="Path to source data (default: %(default)s)",
    )
    build_ks.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Rebuild cache even if a cached index is present",
    )
    # attach handler
    build_ks.set_defaults(func=handle_keyword_search)

    # ________________________________________________________________________________
    # ___________________________semantic search_____________________________________
    # ________________________________________________________________________________
    build_ssc = subparsers.add_parser(
        "semantic_search", help="Use vectorDB to search based on semantics"
    )
    build_ssc.add_argument(
        "query",
        type=str,
        help="Path to source data (default: %(default)s)",
    )
    build_ssc.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Rebuild cache even if a cached index is present",
    )
    # attach handler
    build_ssc.set_defaults(func=handle_semantic_search)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
