# rag_search_engine/utils/keyword_search.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import json
import math

from rag_search_engine.utils.base_search_db import BaseSearchDB
from rag_search_engine.utils.utils import preprocess  # your existing normalizer


class KeywordSearch(BaseSearchDB):
    """
    Keyword / inverted-index search backed by SQLite tables.

    Additional schema:
      - terms(term_id INTEGER PRIMARY KEY, term TEXT UNIQUE)
      - postings(term_id INTEGER, doc_id INTEGER, positions TEXT, PRIMARY KEY(term_id, doc_id))
      - doclen(doc_id INTEGER PRIMARY KEY, length INTEGER)
    """

    TITLE_END_TOKEN = "[TITLE_END]"

    def __init__(self, docs_path: Path | str, db_path: Path | str | None = None) -> None:
        super().__init__(docs_path=docs_path, db_path=db_path)

        self._init_keyword_schema()
        self._build_or_sync_index()

    # ------------------------ schema ------------------------ #
    def _init_keyword_schema(self) -> None:
        cur = self.conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS terms (
                id   INTEGER PRIMARY KEY,
                term TEXT UNIQUE NOT NULL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS postings (
                term_id   INTEGER NOT NULL,
                doc_id    INTEGER NOT NULL,
                positions TEXT NOT NULL,
                PRIMARY KEY (term_id, doc_id),
                FOREIGN KEY (term_id) REFERENCES terms(id) ON DELETE CASCADE,
                FOREIGN KEY (doc_id) REFERENCES movies(id) ON DELETE CASCADE
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS doclen (
                doc_id INTEGER PRIMARY KEY,
                length INTEGER NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES movies(id) ON DELETE CASCADE
            )
            """
        )

        self.conn.commit()

    # ------------------------ build / sync index ------------------------ #
    def _index_doc_count(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM doclen")
        (count,) = cur.fetchone()
        return count

    def _build_or_sync_index(self) -> None:
        """
        Ensure inverted index matches current documents.

        Strategy:
          - If number of entries in doclen != number of docs, rebuild index.
        """
        n_docs = len(self.documents)
        if self._index_doc_count() == n_docs:
            return  # assume synced

        self._rebuild_index()

    def _rebuild_index(self) -> None:
        cur = self.conn.cursor()

        # Clear existing index tables
        cur.execute("DELETE FROM postings")
        cur.execute("DELETE FROM terms")
        cur.execute("DELETE FROM doclen")
        self.conn.commit()

        # Prepare text
        titles = [d["title"] for d in self.documents]
        descriptions = [d["description"] for d in self.documents]

        title_tok_lists = preprocess(titles)
        body_tok_lists = preprocess(descriptions)

        # In-memory structures for one-time build
        term_id_cache: Dict[str, int] = {}
        postings_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        doclen_map: Dict[int, int] = {}

        for d, t_tokens, b_tokens in zip(self.documents, title_tok_lists, body_tok_lists):
            doc_id = int(d["id"])
            tokens = t_tokens + [self.TITLE_END_TOKEN] + b_tokens
            doclen_map[doc_id] = len(tokens)

            for pos, tok in enumerate(tokens):
                # Get/create term_id
                term_id = term_id_cache.get(tok)
                if term_id is None:
                    cur.execute("INSERT INTO terms(term) VALUES (?)", (tok,))
                    term_id = cur.lastrowid
                    term_id_cache[tok] = term_id

                postings_map[(term_id, doc_id)].append(pos)

        # Insert doc lengths
        cur.executemany(
            "INSERT INTO doclen(doc_id, length) VALUES (?, ?)",
            [(doc_id, length) for doc_id, length in doclen_map.items()],
        )

        # Insert postings (positions as JSON)
        posting_rows = []
        for (term_id, doc_id), positions in postings_map.items():
            positions_json = json.dumps(positions)
            posting_rows.append((term_id, doc_id, positions_json))

        cur.executemany(
            "INSERT INTO postings(term_id, doc_id, positions) VALUES (?, ?, ?)",
            posting_rows,
        )

        self.conn.commit()

    # ------------------------ BM25 search ------------------------ #
    def search(
        self, query: str, k: int = 10, k1: float = 1.5, b: float = 0.75
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Simple BM25 search over the DB-backed inverted index.

        Returns list of (doc_id, score, metadata) sorted by score descending.
        """
        # Tokenize query (reuse same pipeline)
        query_tokens = preprocess([query])[0]
        if not query_tokens:
            return []

        cur = self.conn.cursor()

        # Global stats
        N = self.count_movies()
        cur.execute("SELECT AVG(length) FROM doclen")
        (avgdl,) = cur.fetchone()
        if avgdl is None or avgdl == 0:
            return []

        scores: Dict[int, float] = {}
        doclen_cache: Dict[int, int] = {}

        for tok in query_tokens:
            # Find term_id
            cur.execute("SELECT id FROM terms WHERE term = ?", (tok,))
            row = cur.fetchone()
            if not row:
                continue
            term_id = row[0]

            # Fetch postings for this term
            cur.execute(
                "SELECT doc_id, positions FROM postings WHERE term_id = ?",
                (term_id,),
            )
            rows = cur.fetchall()
            if not rows:
                continue

            df = len(rows)
            # BM25 IDF
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id, positions_json in rows:
                positions = json.loads(positions_json)
                tf = len(positions)

                # doc length
                dl = doclen_cache.get(doc_id)
                if dl is None:
                    cur.execute(
                        "SELECT length FROM doclen WHERE doc_id = ?", (doc_id,)
                    )
                    dl_row = cur.fetchone()
                    if not dl_row:
                        continue
                    dl = dl_row[0]
                    doclen_cache[doc_id] = dl

                # BM25 term score
                denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                score_add = idf * (tf * (k1 + 1.0) / denom)

                scores[doc_id] = scores.get(doc_id, 0.0) + score_add

        if not scores:
            return []

        # Sort by score descending and keep top-k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        doc_ids = [doc_id for doc_id, _ in sorted_docs]

        # Fetch metadata
        placeholders = ",".join("?" for _ in doc_ids)
        cur.execute(
            f"SELECT id, title, description FROM movies WHERE id IN ({placeholders})",
            doc_ids,
        )
        meta_rows = {row[0]: (row[1], row[2]) for row in cur.fetchall()}

        results: List[Tuple[int, float, Dict[str, Any]]] = []
        for doc_id, score in sorted_docs:
            title, desc = meta_rows.get(doc_id, ("<missing>", ""))  # should exist
            meta = {"title": title, "description": desc}
            results.append((doc_id, score, meta))

        return results


# ------------------------ Top-level helper for CLI ------------------------ #
def keyword_query(docs_path: str | Path, query: str, k: int = 10) -> None:
    ks = KeywordSearch(docs_path)
    results = ks.search(query, k=k)

    for rank, (doc_id, score, meta) in enumerate(results, start=1):
        title = meta.get("title", f"doc {doc_id}")
        print(f"{rank:2d}. {score:.4f}  {title}")

