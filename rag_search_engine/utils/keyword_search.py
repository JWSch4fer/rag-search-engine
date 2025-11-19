# rag_search_engine/utils/keyword_search.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import json
import math

from rag_search_engine.utils.basesearch_db import BaseSearchDB
from rag_search_engine.utils.utils import preprocess


class KeywordSearch(BaseSearchDB):
    """
    Keyword / inverted-index search backed by SQLite tables.

    Additional schema:
      - terms(term_id INTEGER PRIMARY KEY, term TEXT UNIQUE)
      - postings(term_id INTEGER, doc_id INTEGER, positions TEXT, PRIMARY KEY(term_id, doc_id))
      - doclen(doc_id INTEGER PRIMARY KEY, length INTEGER)
    """

    TITLE_END_TOKEN = "[TITLE_END]"

    def __init__(
        self,
        docs_path: Path | str,
        db_path: Path | str | None = None,
        force: bool = False,
    ) -> None:
        """
        - If docs_path is provided: BaseSearchDB will sync movies, and we will:
          build/sync the inverted index (respecting force).
        - If docs_path is None: open existing DB in query-only mode (no rebuild).
        """
        super().__init__(docs_path=docs_path, db_path=db_path, force=force)
        self._init_keyword_schema()
        if self.docs_path:
            self._build_or_sync_index(force=force)

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

    def _build_or_sync_index(self, force: bool) -> None:
        """
        Ensure inverted index matches current documents.

        Strategy:
          - If number of entries in doclen != number of docs, rebuild index.
        """
        if force:
            self._rebuild_index()

        if self._index_doc_count() == len(self.documents):
            return  # assume synced

        self._rebuild_index()

    def _rebuild_index(self) -> None:
        cur = self.conn.cursor()

        # Clear existing index tables
        cur.execute("DELETE FROM postings")
        cur.execute("DELETE FROM terms")
        cur.execute("DELETE FROM doclen")
        self.conn.commit()

        cur = self.conn.cursor()

        # Prepare text
        titles = [d["title"] for d in self.documents]
        descriptions = [d["description"] for d in self.documents]

        title_tok_lists = preprocess(titles)
        body_tok_lists = preprocess(descriptions)

        # In-memory structures
        doclen_map: Dict[int, int] = {}
        postings_by_term: Dict[str, Dict[int, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        all_terms: set[str] = set()

        # 1) Build doc lengths + postings grouped by *term string* in memory
        for d, t_tokens, b_tokens in zip(
            self.documents, title_tok_lists, body_tok_lists
        ):
            doc_id = int(d["id"])
            tokens = t_tokens + [self.TITLE_END_TOKEN] + b_tokens
            doclen_map[doc_id] = len(tokens)

            for pos, tok in enumerate(tokens):
                postings_by_term[tok][doc_id].append(pos)
                all_terms.add(tok)

        # 2) Load any existing terms from DB
        term_id_cache: Dict[str, int] = {}
        cur.execute("SELECT id, term FROM terms")
        for term_id, term in cur.fetchall():
            term_id_cache[term] = term_id

        # 3) Insert any *new* terms in bulk
        new_terms = [t for t in all_terms if t not in term_id_cache]
        if new_terms:
            cur.executemany(
                "INSERT OR IGNORE INTO terms(term) VALUES (?)",
                [(t,) for t in new_terms],
            )
            # Reload mapping for all terms (simple + robust)
            term_id_cache.clear()
            cur.execute("SELECT id, term FROM terms")
            for term_id, term in cur.fetchall():
                term_id_cache[term] = term_id

        # 4) Insert doc lengths in bulk
        cur.executemany(
            "INSERT INTO doclen(doc_id, length) VALUES (?, ?)",
            [(doc_id, length) for doc_id, length in doclen_map.items()],
        )

        # 5) Build postings rows (term_id, doc_id, positions_json) and bulk insert
        posting_rows: List[tuple[int, int, str]] = []
        for term, doc_map in postings_by_term.items():
            term_id = term_id_cache[term]
            for doc_id, positions in doc_map.items():
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
    ) -> List[Dict[str, Any]]:
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
                    cur.execute("SELECT length FROM doclen WHERE doc_id = ?", (doc_id,))
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

        results: List[Dict[str, Any]] = []
        for doc_id, score in sorted_docs:
            title, desc = meta_rows.get(doc_id, ("<missing>", ""))  # should exist
            meta = {"title": title, "description": desc, "id": doc_id, "score": score}
            results.append(meta)

        return results

    # ------------------------ Verify helpers ------------------------ #
    def verify_db(self) -> None:
        cur = self.conn.cursor()

        # Movies (from BaseSearchDB)
        cur.execute("SELECT COUNT(*) FROM movies")
        (movie_count,) = cur.fetchone()

        # Terms
        cur.execute("SELECT COUNT(*) FROM terms")
        (term_count,) = cur.fetchone()

        # Postings
        cur.execute("SELECT COUNT(*) FROM postings")
        (posting_count,) = cur.fetchone()

        # Doc lengths
        cur.execute("SELECT COUNT(*) FROM doclen")
        (doclen_count,) = cur.fetchone()

        # Avg doc length (BM25 uses this)
        cur.execute("SELECT AVG(length) FROM doclen")
        (avgdl,) = cur.fetchone()
        avgdl = avgdl or 0.0

        print(f"Keyword index DB path:   {self.db_path}")
        print(f"Movies table count:      {movie_count}")
        print(f"Terms table count:       {term_count}")
        print(f"Postings table count:    {posting_count}")
        print(f"Doclen table count:      {doclen_count}")
        print(f"Average doc length:      {avgdl:.2f}")

        # Simple consistency checks
        if movie_count != doclen_count:
            print("WARNING: movies.count != doclen.count (index may be out of sync)")
        if posting_count == 0:
            print("WARNING: postings table is empty")
        if term_count == 0:
            print("WARNING: terms table is empty")
