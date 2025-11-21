# rag_search_engine/utils/hybrid_search.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import time, logging
from sentence_transformers import CrossEncoder

from rag_search_engine.utils.basesearch_db import DEFAULT_DB_PATH
from rag_search_engine.utils.keyword_search import KeywordSearch
from rag_search_engine.utils.semantic_search import SemanticSearch
from rag_search_engine.utils.utils import min_max_norm, rrf_score
from rag_search_engine.llm.gemini import Gemini

logger = logging.getLogger(__name__)


class HybridSearch:
    """
    Hybrid search that combines:

      - KeywordSearch (BM25 over inverted index in SQLite)
      - SemanticSearch (chunked sqlite-vec search over the same movies)

    Both operate over the same SQLite DB file (DEFAULT_DB_PATH by default).
    """

    def __init__(
        self,
        docs_path: Path | str | None = None,
        db_path: Path | str | None = None,
        *,
        force: bool = False,
        max_chunk_size: int = 3,
        overlap: int = 1,
    ) -> None:

        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH

        # BM25 / keyword side
        self.keyword = KeywordSearch(
            docs_path=docs_path,
            db_path=self.db_path,
            force=force,
        )

        # Semantic / chunked side
        self.semantic = SemanticSearch(
            docs_path=docs_path,
            db_path=self.db_path,
            max_chunk_size=max_chunk_size,
            overlap=overlap,
            force=force,
        )

    # ---------------- BM25 wrapper ---------------- #
    def _bm25_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        KeywordSearch.search returns:
          [
            {
              "title": ...,
              "description": ...,
              "id": <doc_id>,
              "score": <bm25_score>,
            },
            ...
          ]
        """
        return self.keyword.search(query, k=limit)

    # ---------------- Semantic wrapper ---------------- #
    def _semantic_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        SemanticSearch.query_top_k returns:
          [
            {
              "chunk_id": ...,
              "distance": <cosine_distance>,
              "chunk": <chunk_text>,
              "movie_id": <doc_id>,
              "title": ...,
              "description": ...,
            },
            ...
          ]
        """
        return self.semantic.query_top_k(query_text=query, k=limit)

    # ---------------- Weighted hybrid search ---------------- #
    def weighted_search(
        self,
        query: str,
        alpha: float,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Weighted hybrid search.

        - Run BM25 and semantic search.
        - Min-max normalize both score sets to [0, 1].
        - Combine per-document score as:
            hybrid_score = alpha * bm25_norm + (1 - alpha) * semantic_norm
          where missing modalities contribute 0 for that part.

        Returns: list of dicts sorted by hybrid_score descending:
          {
            "id":          doc_id,
            "title":       ...,
            "description": ...,
            "bm25":        <normalized bm25 score or 0.0>,
            "semantic":    <normalized semantic score or 0.0>,
            "score":       <hybrid score>,
        }
        """
        # ---------------- BM25 ---------------- #
        bm25_hits = self._bm25_search(query=query, limit=limit)
        bm25_scores = [hit["score"] for hit in bm25_hits]
        bm25_norm = min_max_norm(bm25_scores) if bm25_scores else []

        bm25_by_id: Dict[int, Dict[str, Any]] = {}
        for hit, norm in zip(bm25_hits, bm25_norm):
            doc_id = int(hit["id"])
            bm25_by_id[doc_id] = {
                "id": doc_id,
                "title": hit["title"],
                "description": hit["description"],
                "bm25": float(norm),
            }

        # ---------------- Semantic ---------------- #
        sem_hits = self._semantic_search(query=query, limit=limit)
        # Convert distance -> similarity (larger better) before min-max
        sem_sims = [1.0 - float(hit["distance"]) for hit in sem_hits]
        sem_norm = min_max_norm(sem_sims) if sem_sims else []

        sem_by_id: Dict[int, Dict[str, Any]] = {}
        for hit, norm in zip(sem_hits, sem_norm):
            doc_id = int(hit["movie_id"])
            sem_by_id[doc_id] = {
                "id": doc_id,
                "title": hit["title"],
                "description": hit["description"],
                "semantic": float(norm),
            }

        # ---------------- Combine ---------------- #
        all_ids: Set[int] = set(bm25_by_id.keys()) | set(sem_by_id.keys())
        results: List[Dict[str, Any]] = []

        for doc_id in all_ids:
            bm25_part = bm25_by_id.get(doc_id, {})
            sem_part = sem_by_id.get(doc_id, {})

            title = bm25_part.get("title") or sem_part.get("title") or "<unknown>"
            description = (
                bm25_part.get("description") or sem_part.get("description") or ""
            )

            bm25_score = bm25_part.get("bm25", 0.0)
            sem_score = sem_part.get("semantic", 0.0)

            hybrid_score = alpha * bm25_score + (1.0 - alpha) * sem_score

            results.append(
                {
                    "id": doc_id,
                    "title": title,
                    "description": description,
                    "bm25": bm25_score,
                    "semantic": sem_score,
                    "score": hybrid_score,
                }
            )

        # Sort by combined score descending
        results.sort(key=lambda r: r["score"], reverse=True)

        # You can choose to return only top `limit` docs overall:
        return results[:limit]

    # ---------------- RRF hybrid search ---------------- #
    def rrf_search(
        self,
        query: str,
        k: int = 60,
        limit: int = 10,
        rerank_method: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) hybrid search.

        - Get BM25 hits and semantic hits.
        - Rank each list (BM25 by score desc, semantic by distance asc).
        - RRF score per document:
            RRF(d) = 1 / (k + rank_bm25(d)) + 1 / (k + rank_semantic(d))
          (if doc missing from a list, its rank is treated as large).

        Returns list of dicts sorted by RRF score descending:
          {
            "id":          doc_id,
            "title":       ...,
            "description": ...,
            "score":       <rrf score>,
            "bm25_rank":   <rank or None>,
            "sem_rank":    <rank or None>,
          }
        """
        logger.debug(
            "RRF search starting: query=%r, k=%d, limit=%d, rerank_method=%r",
            query,
            k,
            limit,
            rerank_method,
        )

        # ---------------- BM25 ---------------- #
        bm25_hits = self._bm25_search(query=query, limit=limit)
        # Sort by BM25 score descending
        bm25_hits_sorted = sorted(
            bm25_hits,
            key=lambda h: h["score"],
            reverse=True,
        )
        bm25_rank: Dict[int, int] = {}
        bm25_meta: Dict[int, Dict[str, Any]] = {}
        for rank_idx, hit in enumerate(bm25_hits_sorted):
            doc_id = int(hit["id"])
            bm25_rank[doc_id] = rank_idx
            bm25_meta[doc_id] = hit

        # ---------------- Semantic ---------------- #
        sem_hits = self._semantic_search(query=query, limit=limit)
        # Sort by distance ascending (best match first)
        sem_hits_sorted = sorted(
            sem_hits,
            key=lambda h: h["distance"],
        )
        sem_rank: Dict[int, int] = {}
        sem_meta: Dict[int, Dict[str, Any]] = {}
        for rank_idx, hit in enumerate(sem_hits_sorted):
            doc_id = int(hit["movie_id"])
            sem_rank[doc_id] = rank_idx
            sem_meta[doc_id] = hit

        # ---------------- RRF combine ---------------- #
        NOT_FOUND = 99999  # large rank => almost no contribution
        all_ids: Set[int] = set(bm25_rank.keys()) | set(sem_rank.keys())
        results: List[Dict[str, Any]] = []

        for doc_id in all_ids:
            r_bm25 = bm25_rank.get(doc_id, NOT_FOUND)
            r_sem = sem_rank.get(doc_id, NOT_FOUND)

            score = rrf_score(r_bm25, k) + rrf_score(r_sem, k)

            meta = bm25_meta.get(doc_id) or sem_meta.get(doc_id) or {}
            title = meta.get("title", "<unknown>")
            description = meta.get("description", "")

            results.append(
                {
                    "id": doc_id,
                    "title": title,
                    "description": description,
                    "score": score,
                    "bm25_rank": bm25_rank.get(doc_id),
                    "sem_rank": sem_rank.get(doc_id),
                }
            )

        results.sort(key=lambda r: r["score"], reverse=True)
        logger.debug(
            "RRF base results (pre-rerank, top %d): %s",
            len(results),
            results,
        )
        # ---------------- cross-encoder rerank ---------------- #
        if rerank_method == "cross_encoder":
            logger.info(
                "Reranking top %d results using CrossEncoder 'cross-encoder/ms-marco-TinyBERT-L2-v2'",
                limit,
            )

            # Record original RRF rank and build [query, doc_text] pairs
            pairs = []
            for idx, doc in enumerate(results, start=1):
                doc["rrf_rank"] = idx
                doc_text = (
                    f"{doc.get('title', '')} - "
                    f"{doc.get('document') or doc.get('description', '')}"
                )
                pairs.append([query, doc_text])

            if pairs:
                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                scores = cross_encoder.predict(pairs)
                for doc, score in zip(results, scores):
                    doc["cross_encoder_score"] = float(score)

                # Sort by cross-encoder score (desc), break ties with RRF score
                results.sort(
                    key=lambda d: (d.get("cross_encoder_score", 0.0), d["score"]),
                    reverse=True,
                )
            logger.debug(
                "RRF cross_encoder final results (top %d): %s",
                len(results),
                results,
            )

            return results[:limit]

        # ---------------- LLM-based individual reranking ---------------- #
        if rerank_method == "individual":
            logger.info(
                "Reranking top %d results using individual LLM method...", limit
            )
            gi = Gemini()

            for idx, doc in enumerate(results):
                # Respect rate limits; consider increasing if you hit errors.
                if idx > 0:
                    time.sleep(3)

                rerank_score = gi.rerank_document(query, doc, rerank_method)
                doc["rerank_score"] = rerank_score

            # Sort primarily by rerank_score, break ties by RRF score
            results.sort(
                key=lambda r: (r.get("rerank_score", 0.0), r["score"]),
                reverse=True,
            )
            logger.debug(
                "RRF individual rerank final results (top %d): %s",
                len(results),
                results,
            )
            return results[:limit]

        # ---------------- batch rerank ---------------- #
        if rerank_method == "batch":
            logger.info("Reranking top %d results using batch LLM method...", limit)
            gi = Gemini()
            ranked_ids = gi.rerank_batch(query, results, rerank_method)
            # map id -> rerank_rank (1 = best)
            rank_map: Dict[int, int] = {
                doc_id: idx for idx, doc_id in enumerate(ranked_ids, start=1)
            }

            for doc in results:
                doc_id = doc["id"]
                doc["rerank_rank"] = rank_map.get(doc_id)

            # Sort by rerank_rank (1 is best). Docs not in ranked_ids go last.
            results.sort(
                key=lambda r: (
                    r.get("rerank_rank") is None,
                    r.get("rerank_rank") or 1e9,
                )
            )
            logger.debug(
                "RRF batch rerank final results (top %d): %s",
                len(results),
                results,
            )
            return results[:limit]

        # ---------------- No / unknown rerank method ---------------- #
        if rerank_method is not None:
            logger.warning(
                "Unknown rerank_method=%r, using base RRF only", rerank_method
            )
        logger.debug(
            "RRF final results (no rerank, top %d): %s",
            len(results),
            results,
        )
        return results[:limit]

    # ---------------- Cleanup ---------------- #
    def close(self) -> None:
        self.keyword.close()
        self.semantic.close()
