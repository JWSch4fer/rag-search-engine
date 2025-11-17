#!/usr/bin/env python3

import argparse

from rag_search_engine.utils.semantic_search import (
    SemanticSearch,
    SemanticSearchChunked,
    verify_chunk_embeddings,
    verify_embeddings,
    verify_model,
    vdb_query,
    search_chunked,
)

from rag_search_engine.utils.utils import chunk, semantic_chunk


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


def handle_chunk(args):
    chunked_text = chunk(args.text, args.chunk_size, args.overlap)
    print(f"Chunking {len(args.text.strip())} characters")
    for idx, ct in enumerate(chunked_text):
        print(f"{idx+1}. {' '.join(ct)}")


def handle_semantic_chunk(args):
    chunked_text = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
    print(f"Semantically chunking {len(args.text.strip())} characters")
    for idx, ct in enumerate(chunked_text):
        print(f"{idx+1}. {' '.join(ct)}")


def handle_embed_chunks(args):
    verify_chunk_embeddings(args.path, args.max_chunk_size, args.overlap)


def handle_search_chunked(args):
    search_chunked(args.text, args.limit)


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
    build_s.add_argument(
        "--limit", type=int, default=5, help="limit the number of results"
    )
    # attach handler
    build_s.set_defaults(func=handle_vector_search)
    # ________________________________________________________________________________
    # ________________________chunk text______________________________________________
    # ________________________________________________________________________________
    build_s = subparsers.add_parser(
        "chunk", help="split a piece of text into chunks based on sentences"
    )
    build_s.add_argument("text", type=str, help="text to be split")
    build_s.add_argument(
        "--chunk-size", type=int, default=5, help="limit the words in a chunk"
    )
    build_s.add_argument(
        "--overlap", type=int, default=0, help="allow words to overlap in chunks"
    )
    # attach handler
    build_s.set_defaults(func=handle_chunk)
    # ________________________________________________________________________________
    # ________________________semantic chunk text_____________________________________
    # ________________________________________________________________________________
    build_s = subparsers.add_parser(
        "semantic_chunk", help="split a piece of text into chunks based on sentences"
    )
    build_s.add_argument("text", type=str, help="text to be split")
    build_s.add_argument(
        "--max-chunk-size", type=int, default=4, help="limit the sentences in a chunk"
    )
    build_s.add_argument(
        "--overlap", type=int, default=0, help="allow sentences to overlap in chunks"
    )
    # attach handler
    build_s.set_defaults(func=handle_semantic_chunk)
    # ________________________________________________________________________________
    # ________________________build chunk db_____________________________________
    # ________________________________________________________________________________
    build_ec = subparsers.add_parser(
        "embed_chunks", help="create database for chunked text"
    )
    build_ec.add_argument(
        "--path",
        default="/home/joseph/rag-search-engine/data/movies.json",
        type=str,
        help="data base source",
    )
    build_ec.add_argument(
        "--max-chunk-size", type=int, default=4, help="limit the sentences in a chunk"
    )
    build_ec.add_argument(
        "--overlap", type=int, default=1, help="allow sentences to overlap in chunks"
    )
    # attach handler
    build_ec.set_defaults(func=handle_embed_chunks)
    # ________________________________________________________________________________
    # ________________________search chunk db_____________________________________
    # ________________________________________________________________________________
    build_sc = subparsers.add_parser(
        "search_chunked", help="search database of chunked text"
    )
    build_sc.add_argument(
        "text",
        type=str,
        help="text to search the database",
    )
    build_sc.add_argument(
        "--limit",
        type=int,
        default=10,
        help="text to search the database",
    )

    # attach handler
    build_sc.set_defaults(func=handle_search_chunked)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    # print(sys.argv, args)  # <-- uncomment to debug
    args.func(args)


if __name__ == "__main__":
    main()
