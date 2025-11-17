#!/usr/bin/env python3

import argparse

from rag_search_engine.utils.semantic_search import (
    SemanticSearch,
    verify_embeddings,
    verify_model,
    vdb_query,
)


def handle_verify(args):
    verify_model()


def handle_embed(args):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(args.text)
    print(f"Query: {args.text}")
    print(f"First 5 dimensions: {embedding[0][:5]}")
    print(f"Shape: {embedding.shape[1]}")


def handle_verify_embedding(args):
    verify_embeddings(args.file_path)


def handle_vector_search(args):
    vdb_query(args.query, args.limit)


def make_parser():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    # ________________________________________________________________________________
    # ________________________verify command__________________________________________
    # ________________________________________________________________________________
    build_v = subparsers.add_parser(
        "verify", help="verify semantic search module is loaded"
    )
    # attach handler
    build_v.set_defaults(func=handle_verify)
    # ________________________________________________________________________________
    # ________________________embed text command______________________________________
    # ________________________________________________________________________________
    build_e = subparsers.add_parser("embedquery", help="return text embeddings")
    build_e.add_argument("text", type=str, help="text to embed")
    # attach handler
    build_e.set_defaults(func=handle_embed)
    # ________________________________________________________________________________
    # ________________________verify embeddings________________________________________
    # ________________________________________________________________________________
    build_ve = subparsers.add_parser(
        "verify_embeddings", help="build/load vector database and verify"
    )
    build_ve.add_argument(
        "file_path",
        type=str,
        default="/home/joseph/rag-search-engine/data/movies.json",
        help="path to file to create/load the vector database",
    )
    # attach handler
    build_ve.set_defaults(func=handle_verify_embedding)
    # ________________________________________________________________________________
    # ________________________search vectorDB________________________________________
    # ________________________________________________________________________________
    build_s = subparsers.add_parser("search", help="search vector database")
    build_s.add_argument("query", type=str, help="query text to search")
    build_s.add_argument("--limit", type=int, default=5, help="limit the number of results")
    # attach handler
    build_s.set_defaults(func=handle_vector_search)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    # print(sys.argv, args)  # <-- uncomment to debug
    args.func(args)


if __name__ == "__main__":
    main()
