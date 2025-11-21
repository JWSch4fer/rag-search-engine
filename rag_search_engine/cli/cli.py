#!/usr/bin/env python3
import argparse, logging

from rag_search_engine.llm.gemini import Gemini
from rag_search_engine.utils.semantic_search import SemanticSearch
from rag_search_engine.utils.keyword_search import KeywordSearch
from rag_search_engine.utils.hybrid_search import HybridSearch
from rag_search_engine.config import DEFAULT_DB_PATH
from rag_search_engine.llm.multimodal import (
    verify_image_embedding,
    image_search_command,
)

logger = logging.getLogger(__name__)


def handle_image_search(args: argparse.Namespace) -> None:
    results = image_search_command(args.image_path, limit=5)

    for idx, doc in enumerate(results, start=1):
        title = str(doc.get("title", "")).strip()
        sim = float(doc.get("similarity", 0.0))
        desc = str(doc.get("description", "")).strip()

        # Truncate description a bit for display
        preview = (desc[:120] + "…") if len(desc) > 120 else desc

        print(f"{idx}. {title} (similarity: {sim:.3f})")
        print(f"   {preview}")
        print()


def handle_verify_image_embedding(args: argparse.Namespace) -> None:
    """
    CLI handler to verify image embeddings via MultimodalSearch.

    Usage:
        rag-search verify_image_embedding path/to/image.jpeg
    """
    verify_image_embedding(args.image_path)


def handle_multimodal_rewrite(args: argparse.Namespace) -> None:
    """
    Multimodal query rewriting using Gemini: image + text -> rewritten text query.

    Required args:
      --image : path to an image file
      --query : original text query to rewrite
    """
    image_path = args.image
    query = args.query

    logger.info(
        "multimodal_rewrite command starting",
        extra={
            "image_path": image_path,
            "query_preview": query[:100],
        },
    )

    gi = Gemini()

    rewritten, total_tokens = gi.rewrite_multimodal_query(image_path, query)

    print(f"Rewritten query: {rewritten}")
    if total_tokens is not None:
        print(f"Total tokens:    {total_tokens}")


def handle_question(args: argparse.Namespace) -> None:
    """
    'question' command:
    - Load the movies dataset and set up HybridSearch.
    - Perform an RRF search using the provided question.
    - Use Gemini to answer the question based on the search results.
    - Print search results and the answer.
    """
    question = args.question
    limit = args.limit

    logger.info(
        "question command starting",
        extra={
            "question": question,
            "limit": limit,
        },
    )

    # Set up HybridSearch with the movies dataset
    hs = HybridSearch(docs_path=None, db_path=DEFAULT_DB_PATH)
    try:
        hits = hs.rrf_search(
            query=question,
            k=60,
            limit=limit,
            rerank_method="cross_encoder",
        )
    finally:
        hs.close()

    if not hits:
        print("No search results found.")
        return

    # Titles for printing
    titles = [str(h.get("title", "")).strip() for h in hits if h.get("title")]
    logger.debug(
        "question search results titles: %s",
        titles,
    )

    # Build context string for Gemini
    # e.g. "- Jurassic Park: dinosaurs in a theme park..."
    context_lines = []
    for h in hits:
        title = str(h.get("title", "")).strip()
        desc = str(h.get("description", h.get("document", ""))).strip()
        context_lines.append(f"- {title}: {desc}")
    context_str = "\n".join(context_lines)

    # Ask Gemini to answer the question based on these movies
    gi = Gemini()
    answer = gi.answer_question(question, context_str)

    # Print in requested format
    print("Search Results:")
    for t in titles:
        print(f"  - {t}")
    print()
    print("Answer:")
    print(answer)


def handle_citations(args: argparse.Namespace) -> None:
    """
    'citations' command:
    - Load the movies dataset and set up HybridSearch.
    - Perform an RRF search with the provided query.
    - Use Gemini to answer the query with inline citations.
    - Print search results and the LLM answer.
    """
    query = args.query
    limit = args.limit

    logger.info(
        "citations command starting",
        extra={
            "query": query,
            "limit": limit,
        },
    )

    # Set up HybridSearch using the movies dataset + default DB path
    hs = HybridSearch(docs_path=None, db_path=DEFAULT_DB_PATH)
    try:
        hits = hs.rrf_search(
            query=query,
            k=60,
            limit=limit,
            rerank_method="cross_encoder",
        )
    finally:
        hs.close()

    if not hits:
        print("No search results found.")
        return

    # Prepare titles for printing
    titles = [str(h.get("title", "")).strip() for h in hits if h.get("title")]
    logger.debug(
        "citations search results titles: %s",
        titles,
    )

    # Prepare a documents string for the LLM.
    # Pattern: numbered sources [1], [2], etc. to line up with citations.
    documents_lines = []
    for idx, h in enumerate(hits, start=1):
        title = str(h.get("title", "")).strip()
        desc = str(h.get("description", h.get("document", ""))).strip()
        documents_lines.append(f"[{idx}] {title}: {desc}")
    documents_str = "\n".join(documents_lines)

    # Call Gemini to generate an answer with citations
    gi = Gemini()
    answer = gi.answer_with_citations(query, documents_str)

    # Print in requested format
    print("Search Results:")
    for t in titles:
        print(f"  - {t}")
    print()
    print("LLM Answer:")
    print(answer)


def handle_summarize(args: argparse.Namespace):
    """
    Summarize command:
    - Load movies search via HybridSearch.
    - Perform RRF search with the given query.
    - Ask Gemini to summarize the results.
    - Print search results and the LLM summary.
    """
    query = args.query
    limit = args.limit

    logger.info(
        "summarize command starting",
        extra={
            "query": query,
            "limit": limit,
        },
    )

    # Set up HybridSearch using the existing DB
    hs = HybridSearch(docs_path=None, db_path=DEFAULT_DB_PATH)
    try:
        hits = hs.rrf_search(
            query=query,
            k=60,
            limit=limit,
            rerank_method="cross_encoder",
        )
    finally:
        hs.close()

    if not hits:
        print("No search results found.")
        return

    # Prepare titles for printing
    titles = [str(h.get("title", "")).strip() for h in hits if h.get("title")]
    logger.debug(
        "summarize search results titles: %s",
        titles,
    )

    # Prepare rich results string for the LLM
    results_lines = []
    for h in hits:
        title = str(h.get("title", "")).strip()
        desc = str(h.get("description", h.get("document", ""))).strip()
        results_lines.append(f"- {title}: {desc}")
    results_str = "\n".join(results_lines)

    # Call Gemini to summarize results
    gi = Gemini()
    summary = gi.summarize_results(query, results_str)

    # Print in requested format
    print("Search Results:")
    for t in titles:
        print(f"  - {t}")
    print()
    print("LLM Summary:")
    print(summary)


def handle_hybrid_weight(args: argparse.Namespace):
    hs = HybridSearch(
        docs_path=None,
        db_path=DEFAULT_DB_PATH,
    )
    hits = hs.weighted_search(args.query, alpha=args.alpha, limit=args.limit)

    for h in hits:
        print(f"{h['score']:.4f}  {h['title']}")
    hs.close()


def handle_rag(args: argparse.Namespace):
    """
    RAG-style command:
    - Run RRF search over movies for the query.
    - Take top 5 results.
    - Ask Gemini to answer based on those documents.
    - Print search results and the generated answer.
    """
    query = args.query

    # RRF search over existing DB
    hs = HybridSearch(docs_path=None, db_path=DEFAULT_DB_PATH)
    try:
        hits = hs.rrf_search(
            query=query,
            k=60,
            limit=5,
            rerank_method=None,  # pure RRF; adjust if you ever want cross_encoder here
        )
    finally:
        hs.close()

    # Extract titles for display
    titles = [str(h.get("title", "")).strip() for h in hits if h.get("title")]

    # Build a docs string for the LLM (include titles + descriptions)
    docs_lines = []
    for h in hits:
        title = str(h.get("title", "")).strip()
        desc = str(h.get("description", h.get("document", ""))).strip()
        docs_lines.append(f"- {title}: {desc}")
    docs_str = "\n".join(docs_lines)

    # Call Gemini RAG helper
    gi = Gemini()
    answer = gi.rag_answer(query, docs_str)

    # Print in the requested format
    print("Search Results:")
    for t in titles:
        print(f"  - {t}")
    print()
    print("RAG Response:")
    print(answer)


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
    # ________________________________________________________________________________
    # _______________________________rag parser_______________________________________
    # ________________________________________________________________________________

    build_rag = subparsers.add_parser("rag", help="RAG-style answer using RRF + Gemini")
    build_rag.add_argument("query", type=str, help="Search query")
    build_rag.set_defaults(func=handle_rag)

    # ________________________________________________________________________________
    # _______________________________rag summarize____________________________________
    # ________________________________________________________________________________
    build_summary = subparsers.add_parser(
        "summarize",
        help="Summarize top search results for a query using RRF + Gemini",
    )
    build_summary.add_argument(
        "query",
        type=str,
        help="Search query to summarize results for",
    )
    build_summary.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of top results to retrieve and summarize (default: 5)",
    )
    build_summary.set_defaults(func=handle_summarize)
    # ________________________________________________________________________________
    # _______________________________rag summarize____________________________________
    # ________________________________________________________________________________
    build_summary = subparsers.add_parser(
        "citations",
        help="Answer a query with citations based on RRF search results",
    )
    build_summary.add_argument(
        "query",
        type=str,
        help="Search query to answer with citations",
    )
    build_summary.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of top results to retrieve (default: 5)",
    )
    build_summary.set_defaults(func=handle_citations)
    # ________________________________________________________________________________
    # _______________________________rag question/answer______________________________
    # ________________________________________________________________________________
    build_quesiton = subparsers.add_parser(
        "question",
        help="Ask a natural-language question answered using movie search results",
    )
    build_quesiton.add_argument(
        "question",
        type=str,
        help="User question to answer based on Hoopla movie catalog",
    )
    build_quesiton.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of top results to retrieve (default: 5)",
    )
    build_quesiton.set_defaults(func=handle_question)
    # ________________________________________________________________________________
    # _______________________________rag multimodal image_____________________________
    # ________________________________________________________________________________
    build_mm_image = subparsers.add_parser(
        "multimodal_image",
        help="Rewrite a text query using an image and Gemini (multimodal)",
    )
    build_mm_image.add_argument(
        "--image",
        required=True,
        help="Path to the image file (e.g., data/paddington.jpeg)",
    )
    build_mm_image.add_argument(
        "--query",
        required=True,
        help="Original text query to rewrite based on the image",
    )
    build_mm_image.set_defaults(func=handle_multimodal_rewrite)
    # ________________________________________________________________________________
    # _______________________________rag multimodal image_____________________________
    # ________________________________________________________________________________
    verify_img_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Generate an image embedding (CLIP) and print its dimensionality",
    )
    verify_img_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file (e.g., data/paddington.jpeg)",
    )
    verify_img_parser.set_defaults(func=handle_verify_image_embedding)
    # ________________________________________________________________________________
    # _______________________________rag multimodal search_____________________________
    # ________________________________________________________________________________
    cmd_search = subparsers.add_parser(
        "image_search",
        help="Search the movie dataset using an image (CLIP-based similarity)",
    )
    cmd_search.add_argument(
        "image_path",
        type=str,
        help="Path to the image file (e.g., data/paddington.jpeg)",
    )
    cmd_search.set_defaults(func=handle_image_search)
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
