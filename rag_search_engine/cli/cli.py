#!/usr/bin/env python3
import argparse, logging

from rag_search_engine.llm.gemini import Gemini
from rag_search_engine.utils.semantic_search import SemanticSearch
from rag_search_engine.utils.keyword_search import KeywordSearch
from rag_search_engine.utils.hybrid_search import HybridSearch
from rag_search_engine.config import DEFAULT_DB_PATH

logger = logging.getLogger(__name__)


def handle_hybrid_weight(args: argparse.Namespace):
    hs = HybridSearch(
        docs_path=None,
        db_path=DEFAULT_DB_PATH,
    )
    hits = hs.weighted_search(args.query, alpha=args.alpha, limit=args.limit)

    for h in hits:
        print(f"{h['score']:.4f}  {h['title']}")
    hs.close()


def handle_hybrid_rrf(args: argparse.Namespace):
    # Log the original query before any enhancement
    logger.debug("RRF search CLI: original query=%r", args.query)

    # Enhance the query first (no-op if enhance is None)
    gi = Gemini()
    query = gi.enhance(args.enhance, args.query)

    # Log the enhanced query
    logger.debug(
        "RRF search CLI: enhanced query=%r using method=%r",
        query,
        args.enhance,
    )

    # When using any rerank method, gather more candidates for stronger reranking
    gather_limit = args.limit * 2 if args.rerank_method else args.limit

    hs = HybridSearch(
        docs_path=None,
        db_path=DEFAULT_DB_PATH,
    )

    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

    # Let HybridSearch handle all reranking logic, including cross_encoder
    hits = hs.rrf_search(
        query,
        k=args.k,
        limit=gather_limit,
        rerank_method=args.rerank_method,
    )

    # If we did any reranking, only show the top user-specified limit
    if args.rerank_method in ("individual", "batch", "cross_encoder"):
        hits = hits[: args.limit]

        for i, doc in enumerate(hits, start=1):
            line = f"{i}. {doc['title']}"

            # LLM (Gemini) individual/batch rerank score
            if args.rerank_method in ("individual", "batch") and "rerank_score" in doc:
                line += f"\n   Rerank Score: {doc['rerank_score']:.3f}/10"

            # Cross-encoder score if present
            if args.rerank_method == "cross_encoder" and "cross_encoder_score" in doc:
                line += f"\n   Cross-Encoder Score: {doc['cross_encoder_score']:.4f}"

            # RRF + underlying ranks (these should be set in HybridSearch)
            if "score" in doc:
                line += f"\n   RRF Score: {doc['score']:.4f}"
            if "rrf_rank" in doc:
                line += f"\n   RRF Rank: {doc['rrf_rank']}"
            line += (
                f"\n   BM25 Rank: {doc.get('bm25_rank')}, "
                f"Semantic Rank: {doc.get('sem_rank')}"
            )

            line += f"\n   {doc.get('description', '')[:80]}..."
            print(line)
            print()

    else:
        # No reranking: just print the base RRF results
        for h in hits[: args.limit]:
            print(f"{h['score']:.4f}  {h['title']}")

        # ----- Optional LLM evaluation (0–3) -----
    if args.evaluate and hits:
        # Prepare docs for evaluation: only title / description are needed
        eval_docs = [
            {
                "title": d.get("title", ""),
                "description": d.get("description", d.get("document", "")),
            }
            for d in hits
        ]

        scores = gi.evaluate_results(query, eval_docs)

        # Pair scores with results and print the final evaluation report
        print()  # blank line to separate from search results
        for idx, (doc, score) in enumerate(zip(hits, scores), start=1):
            # Clamp to valid range just in case
            s = max(0, min(3, int(score)))
            print(f"{idx}. {doc.get('title', '')}: {s}/3")

    hs.close()


def handle_build(args: argparse.Namespace) -> None:
    # Build DB for keyword search
    ks = KeywordSearch.build_from_docs(
        docs_path=args.file_path, db_path=DEFAULT_DB_PATH, force=args.force
    )
    ks.verify_db()
    ks.close()
    print("--------------------------------------------------------------")
    # Build DB for semantic search
    ssc = SemanticSearch.build_from_docs(
        docs_path=args.file_path, db_path=DEFAULT_DB_PATH, force=args.force
    )
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
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

    # ________________________________________________________________________________
    # ___________________________hybrid search________________________________________
    # ________________________________________________________________________________
    build_ws = subparsers.add_parser(
        "weighted-search",
        help="combine weighted semantic search results with keyword results",
    )
    build_ws.add_argument(
        "query",
        type=str,
        help="Path to source data (default: %(default)s)",
    )
    build_ws.add_argument(
        "--limit",
        type=int,
        default=5,
        help="limit the number of search results returned",
    )
    build_ws.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="change the weight of semantic vs keyword results (larger value leans toward keyword)",
    )
    # attach handler
    build_ws.set_defaults(func=handle_hybrid_weight)

    # ________________________________________________________________________________
    # ___________________________hybrid search rrfl___________________________________
    # ________________________________________________________________________________
    build_rrf = subparsers.add_parser(
        "rrf-search",
        help="combine weighted semantic search results with keyword results",
    )
    build_rrf.add_argument(
        "query",
        type=str,
        help="Path to source data (default: %(default)s)",
    )
    build_rrf.add_argument(
        "--limit",
        type=int,
        default=5,
        help="limit the number of search results returned",
    )
    build_rrf.add_argument(
        "--k",
        type=float,
        default=60,
        help="adjust combination ranking from semantic+keyword search",
    )
    build_rrf.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand", None],
        default=None,
        help="Query enhancement method with Gemini",
    )
    build_rrf.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder", None],
        default=None,
        help="rerank results with Gemini",
    )  # attach handler
    build_rrf.add_argument(
    "--evaluate",
    action="store_true",
    help="Use an LLM to rate each result from 0-3 for relevance",
    )
    build_rrf.set_defaults(func=handle_hybrid_rrf)
    return parser


def setup_logging(debug: bool = False) -> None:
    """
    Configure root logging.

    If `debug` is True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    logger.setLevel(level)


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    setup_logging(debug=args.debug)
    args.func(args)


if __name__ == "__main__":
    main()
