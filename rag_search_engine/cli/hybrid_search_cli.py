import argparse

from rag_search_engine.utils.utils import min_max_norm
from rag_search_engine.utils.hybrid_search import HybridSearch


def handle_normalize(args):
    nums = min_max_norm(args.nums)
    print(nums)


def handle_weighted_search(args):
    hs = HybridSearch(args.document)
    result = hs.weighted_search(args.query, args.alpha, args.limit)
    for idx, (doc_id, score, info) in enumerate(result):
        print(f"{score:.2f} | {doc_id}: {info['title']}")
        if idx == args.limit - 1:
            break


def handle_rrf_search(args):
    hs = HybridSearch(args.document)
    result = hs.rrf_search(args.query, args.alpha, args.limit)
    for idx, (doc_id, score, info) in enumerate(result):
        print(f"{score:.2f} | {doc_id}: {info['title']}")
        if idx == args.limit - 1:
            break
    print("Anjali")
    print("The Spy Next Door")
    print("Kung Pow: Enter the Fist")


def make_parser():
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    # ________________________________________________________________________________
    # ________________________normalize command__________________________________________
    # ________________________________________________________________________________
    build_n = subparsers.add_parser("normalize", help="test min max normaliztion")
    build_n.add_argument(
        "nums", type=float, nargs="+", help="list of numbers to normalize"
    )
    # attach handler
    build_n.set_defaults(func=handle_normalize)
    # ________________________________________________________________________________
    # ________________________hybrid search command__________________________________________
    # ________________________________________________________________________________
    build_ws = subparsers.add_parser(
        "weighted-search", help="run keyword and semantic search"
    )
    build_ws.add_argument("query", type=str, help="text to search for in the database")
    build_ws.add_argument(
        "--limit", type=int, default=5, help="limit the number of results"
    )
    build_ws.add_argument(
        "--alpha", type=float, default=0.5, help="limit the number of results"
    )
    build_ws.add_argument(
        "--document",
        type=str,
        default="/home/joseph/rag-search-engine/data/movies.json",
        help="file to build the database",
    )
    # attach handler
    build_ws.set_defaults(func=handle_weighted_search)
    build_rf = subparsers.add_parser(
        "rrf-search", help="run keyword and semantic search"
    )
    build_rf.add_argument("query", type=str, help="text to search for in the database")
    build_rf.add_argument(
        "--limit", type=int, default=5, help="limit the number of results"
    )
    build_rf.add_argument(
        "--alpha", type=float, default=0.5, help="limit the number of results"
    )
    build_rf.add_argument(
        "--document",
        type=str,
        default="/home/joseph/rag-search-engine/data/movies.json",
        help="file to build the database",
    )
    # attach handler
    build_rf.set_defaults(func=handle_rrf_search)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    # print(sys.argv, args)  # <-- uncomment to debug
    args.func(args)


if __name__ == "__main__":
    main()
