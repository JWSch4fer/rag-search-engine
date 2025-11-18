# rag_search_engine/utils/semantic_search.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

# import sqlite3
import numpy as np
import numpy.typing as npt
import sqlite_vec
from sentence_transformers import SentenceTransformer

from rag_search_engine.utils.basesearch_db import BaseSearchDB
from rag_search_engine.utils.utils import semantic_chunk


class SemanticSearch(BaseSearchDB):
    """
    sqlite-vec–backed semantic search over chunked movie docs.

    Schema additions (on top of BaseSearchDB.movies):

      - chunk_movies(
            id          INTEGER PRIMARY KEY,  -- same as movies.id
            title       TEXT NOT NULL,
            description TEXT NOT NULL
        )

      - chunks(
            id       INTEGER PRIMARY KEY,
            movie_id INTEGER NOT NULL,        -- FK to chunk_movies.id
            chunk    TEXT NOT NULL,
            FOREIGN KEY (movie_id) REFERENCES chunk_movies(id) ON DELETE CASCADE
        )

      - chunk_embeddings (vec0 virtual table):
            embedding float[dim] distance_metric=cosine
        rowid of chunk_embeddings == chunks.id
    """

    def __init__(
        self,
        docs_path: Path | str,
        db_path: Path | str | None = None,
        force: bool = False,
        max_chunk_size: int = 3,
        overlap: int = 1,
    ) -> None:
        # Local embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Initialize shared DB + movies table
        super().__init__(docs_path=docs_path, db_path=db_path, force=force)

        # sqlite-vec + chunk-specific schema
        self._chunk_load_vec_extension()
        self._chunk_init_schema()
        if self.docs_path:
            self._build_or_sync_chunk(max_chunk_size=max_chunk_size, overlap=overlap)

    # ------------------------ SQLite / schema ------------------------ #
    def _chunk_load_vec_extension(self) -> None:
        """Load sqlite-vec into this SQLite connection."""
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)  # loads vec_version(), vec0, etc.
        self.conn.enable_load_extension(False)

    def _chunk_init_schema(self) -> None:
        """Create chunk tables + vec0 virtual table if they don't exist."""
        cur = self.conn.cursor()

        # One row per movie (mirrors movies table, but kept separate for chunked search)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_movies (
                id          INTEGER PRIMARY KEY,
                title       TEXT NOT NULL,
                description TEXT NOT NULL
            )
            """
        )

        # Many chunks per movie
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id       INTEGER PRIMARY KEY,
                movie_id INTEGER NOT NULL,
                chunk    TEXT NOT NULL,
                FOREIGN KEY (movie_id) REFERENCES chunk_movies(id)
                    ON DELETE CASCADE
            )
            """
        )

        # vec0 table storing embeddings for each chunk; cosine distance for semantic similarity
        cur.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings
            USING vec0(
              embedding float[{self.embedding_dim}] distance_metric=cosine
            )
            """
        )

        self.conn.commit()

    # ------------------------ Build / Sync ------------------------ #
    def _build_or_sync_chunk(
        self,
        max_chunk_size: int,
        overlap: int,
    ) -> None:
        """
        Ensure the chunked DB matches self.documents.

        Strategy:
          - If chunk_movies count != number of documents, wipe + rebuild
            chunk_movies, chunks, and chunk_embeddings.
        """
        n_docs = len(self.documents)
        cur = self.conn.cursor()

        cur.execute("SELECT COUNT(*) FROM chunk_movies")
        row = cur.fetchone()
        existing = row[0] if row else 0

        if existing == n_docs:
            # Simple heuristic: assume DB already matches docs
            return

        # Rebuild from scratch
        cur.execute("DELETE FROM chunk_embeddings")
        cur.execute("DELETE FROM chunks")
        cur.execute("DELETE FROM chunk_movies")
        self.conn.commit()

        chunk_rows: List[tuple[int, int, str]] = []
        chunk_id = 0

        # Insert movies + prepare chunk rows
        for d in self.documents:
            doc_id = int(d["id"])
            title = d["title"]
            description = d["description"]

            # Insert into chunk_movies (id matches movies.id)
            cur.execute(
                "INSERT INTO chunk_movies (id, title, description) VALUES (?, ?, ?)",
                (doc_id, title, description),
            )

            # Title as its own chunk
            chunk_rows.append((chunk_id, doc_id, title))
            chunk_id += 1

            # Description chunks
            for c_tokens in semantic_chunk(description, max_chunk_size, overlap):
                # semantic_chunk returns list[str]; join to a single chunk string
                chunk_text = "".join(c_tokens)
                chunk_rows.append((chunk_id, doc_id, chunk_text))
                chunk_id += 1

        # Insert all chunks (chunks.id == chunk_embeddings.rowid)
        cur.executemany(
            "INSERT INTO chunks (id, movie_id, chunk) VALUES (?, ?, ?)",
            chunk_rows,
        )

        # Embed all chunks
        chunk_ids = [row[0] for row in chunk_rows]
        chunk_texts = [row[2] for row in chunk_rows]

        embeddings = self.model.encode(chunk_texts, show_progress_bar=True)
        emb_arr: npt.NDArray[np.float32] = np.asarray(embeddings, dtype=np.float32)

        cur.executemany(
            "INSERT INTO chunk_embeddings(rowid, embedding) VALUES (?, ?)",
            list(zip(chunk_ids, emb_arr)),
        )

        self.conn.commit()

    # ------------------------ Embeddings ------------------------ #
    def generate_embedding(self, text: str | List[str]) -> npt.NDArray[np.float32]:
        """Embed a single string or list of strings."""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        if any(len(t.strip()) == 0 for t in texts):
            raise ValueError("cannot embed empty text")

        arr = self.model.encode(texts)
        return np.asarray(arr, dtype=np.float32)

    # ------------------------ Query ------------------------ #
    def query_top_k(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Return top-k results as a list of dicts:

          {
            "chunk_id": rowid,
            "distance": distance,
            "chunk": chunk_text,
            "movie_id": movie_id,
            "title": title,
            "description": full_description,
          }

        We use:
          - vec0 KNN: WHERE embedding MATCH :q AND k = :k
          - distance_metric=cosine, so sqlite-vec gives us cosine *distance*
        """
        query_vec = self.generate_embedding(query_text)[0]

        cur = self.conn.cursor()
        cur.execute(
            """
            WITH knn AS (
                SELECT rowid, distance
                FROM chunk_embeddings
                WHERE embedding MATCH :q
                  AND k = :k
            )
            SELECT
                knn.rowid,
                knn.distance,
                c.chunk,
                m.id,
                m.title,
                m.description
            FROM knn
            JOIN chunks AS c
              ON c.id = knn.rowid
            JOIN chunk_movies AS m
              ON m.id = c.movie_id
            ORDER BY knn.distance ASC
            """,
            {"q": query_vec, "k": int(k)},
        )

        rows = cur.fetchall()

        # Aggregate: keep best (lowest distance) chunk per movie
        best_by_movie: Dict[int, Dict[str, Any]] = {}

        for rowid, distance, chunk, movie_id, title, description in rows:
            movie_id = int(movie_id)
            distance = float(distance)
            prev = best_by_movie.get(movie_id)

            if prev is None or distance < prev["distance"]:
                best_by_movie[movie_id] = {
                    "chunk_id": int(rowid),
                    "distance": distance,
                    "chunk": chunk,
                    "movie_id": movie_id,
                    "title": title,
                    "description": description,
                }

        # Sort movies by their best distance (ascending) and take top-k
        deduped = sorted(
            best_by_movie.values(),
            key=lambda r: r["distance"],
        )

        return deduped[:k]

