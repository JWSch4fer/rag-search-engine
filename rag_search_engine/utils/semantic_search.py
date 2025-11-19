# rag_search_engine/utils/semantic_search.py
# rag_search_engine/utils/semantic_search.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

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

      - chunks(
            id             INTEGER PRIMARY KEY,
            movie_id       INTEGER NOT NULL,   -- FK to movies.id
            chunk_index    INTEGER NOT NULL,   -- 0 = title, 1..N = desc chunks
            max_chunk_size INTEGER NOT NULL,
            overlap        INTEGER NOT NULL,
            FOREIGN KEY (movie_id) REFERENCES movies(id) ON DELETE CASCADE
        )

      - chunk_embeddings (vec0 virtual table):
            embedding float[dim] distance_metric=cosine
        rowid of chunk_embeddings == chunks.id
    """

    def __init__(
        self,
        docs_path: Path | str | None = None,
        db_path: Path | str | None = None,
        max_chunk_size: int = 3,
        overlap: int = 1,
        force: bool = False,
    ) -> None:
        # Local embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Save parameters (used for build + reconstruction)
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

        # Initialize shared DB + movies table (if docs_path given)
        super().__init__(docs_path=docs_path, db_path=db_path, force=force)

        # sqlite-vec + chunk-specific schema
        self._chunk_load_vec_extension()
        self._chunk_init_schema()

        # Only build/sync chunks when we actually have docs (i.e. build mode)
        if docs_path is not None:
            self._build_or_sync_chunk(
                max_chunk_size=max_chunk_size,
                overlap=overlap,
                force=force,
            )

    # ------------------------ SQLite / schema ------------------------ #
    def _chunk_load_vec_extension(self) -> None:
        """Load sqlite-vec into this SQLite connection."""
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)  # loads vec_version(), vec0, etc.
        self.conn.enable_load_extension(False)

    def _chunk_init_schema(self) -> None:
        """Create chunk tables + vec0 virtual table if they don't exist."""
        cur = self.conn.cursor()

        # Only chunks; movies table is created in BaseSearchDB
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id             INTEGER PRIMARY KEY,
                movie_id       INTEGER NOT NULL,
                chunk_index    INTEGER NOT NULL,   -- 0 = title, 1..N = desc chunks
                max_chunk_size INTEGER NOT NULL,
                overlap        INTEGER NOT NULL,
                FOREIGN KEY (movie_id) REFERENCES movies(id)
                    ON DELETE CASCADE
            )
            """
        )

        # vec0 table storing embeddings for each chunk; cosine distance
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
        force: bool = False,
    ) -> None:
        """
        Ensure the chunk index + embeddings match the movies table and the
        given chunking parameters.

        Strategy:
          - If force=True, always rebuild.
          - Otherwise, skip rebuild if:
              * number of distinct movies in chunks == number of movies
              * and all rows share the same (max_chunk_size, overlap) pair
              * and chunk_embeddings count == chunks count
        """
        cur = self.conn.cursor()

        # Count movies
        cur.execute("SELECT COUNT(*) FROM movies")
        (movie_count,) = cur.fetchone()

        # Count chunks and movies represented there
        cur.execute("SELECT COUNT(*) FROM chunks")
        (chunk_count,) = cur.fetchone()

        cur.execute("SELECT COUNT(DISTINCT movie_id) FROM chunks")
        (chunk_movie_count,) = cur.fetchone()

        # Check chunk params
        cur.execute("SELECT DISTINCT max_chunk_size, overlap FROM chunks")
        params = cur.fetchall()  # list of (max_chunk_size, overlap)

        # Check embeddings count
        cur.execute("SELECT COUNT(*) FROM chunk_embeddings")
        (emb_count,) = cur.fetchone()

        # Decide if we can skip rebuild
        if not force:
            in_sync = (
                chunk_count > 0
                and chunk_movie_count == movie_count
                and len(params) == 1
                and params[0] == (max_chunk_size, overlap)
                and emb_count == chunk_count
            )
            if in_sync:
                return

        # Rebuild from scratch: only chunks + embeddings
        cur.execute("DELETE FROM chunk_embeddings")
        cur.execute("DELETE FROM chunks")
        self.conn.commit()

        chunk_rows: List[tuple[int, int, int, int, int]] = []
        chunk_ids: List[int] = []
        chunk_texts: List[str] = []
        chunk_id = 0

        # Read movies from DB; BaseSearchDB already synced this
        cur.execute("SELECT id, title, description FROM movies ORDER BY id")
        movies = cur.fetchall()

        for movie_id, title, description in movies:
            movie_id = int(movie_id)

            # Title as chunk_index 0
            chunk_rows.append((chunk_id, movie_id, 0, max_chunk_size, overlap))
            chunk_ids.append(chunk_id)
            chunk_texts.append(title)
            chunk_id += 1

            # Description chunks: indices 1..N
            desc_chunks = semantic_chunk(description, max_chunk_size, overlap)
            for local_idx, c_tokens in enumerate(desc_chunks, start=1):
                chunk_text = "".join(c_tokens)
                chunk_rows.append(
                    (chunk_id, movie_id, local_idx, max_chunk_size, overlap)
                )
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk_text)
                chunk_id += 1

        # Insert all chunks (chunks.id == chunk_embeddings.rowid)
        cur.executemany(
            """
            INSERT INTO chunks (id, movie_id, chunk_index, max_chunk_size, overlap)
            VALUES (?, ?, ?, ?, ?)
            """,
            chunk_rows,
        )

        # Embed all chunks
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
    def query_top_k(
        self,
        query_text: str,
        k: int = 5,
        knn_multiplier: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Return top-k *movies*, each represented by its best-matching chunk.

        Strategy:
          - Ask sqlite-vec for k * knn_multiplier chunk neighbors.
          - For each movie_id, keep only the chunk with the smallest distance.
          - Reconstruct that chunk text on the fly from (title/description, chunk_index, max_chunk_size, overlap).
          - Sort movies by best distance and return at most k.

        Returned dicts look like:
          {
            "chunk_id": rowid,
            "distance": distance,
            "chunk": chunk_text,
            "movie_id": movie_id,
            "title": title,
            "description": full_description,
          }
        """
        query_vec = self.generate_embedding(query_text)[0]
        internal_k = max(k * knn_multiplier, k)

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
                c.movie_id,
                c.chunk_index,
                c.max_chunk_size,
                c.overlap,
                m.title,
                m.description
            FROM knn
            JOIN chunks AS c
              ON c.id = knn.rowid
            JOIN movies AS m
              ON m.id = c.movie_id
            ORDER BY knn.distance ASC
            """,
            {"q": query_vec, "k": int(internal_k)},
        )

        rows = cur.fetchall()

        # Aggregate: keep best (lowest distance) chunk per movie
        # Store minimal info; reconstruct chunk text later to avoid extra work
        best_by_movie: Dict[int, Dict[str, Any]] = {}

        for (
            rowid,
            distance,
            movie_id,
            chunk_index,
            max_chunk_size,
            overlap,
            title,
            description,
        ) in rows:
            movie_id = int(movie_id)
            distance = float(distance)
            prev = best_by_movie.get(movie_id)

            if prev is None or distance < prev["distance"]:
                best_by_movie[movie_id] = {
                    "chunk_id": int(rowid),
                    "distance": distance,
                    "movie_id": movie_id,
                    "chunk_index": int(chunk_index),
                    "max_chunk_size": int(max_chunk_size),
                    "overlap": int(overlap),
                    "title": title,
                    "description": description,
                }

        # Sort movies by their best distance
        movie_hits = sorted(
            best_by_movie.values(),
            key=lambda r: r["distance"],
        )[:k]

        # Reconstruct chunk text for the final hits
        results: List[Dict[str, Any]] = []
        for hit in movie_hits:
            chunk_text = self._reconstruct_chunk_text(
                title=hit["title"],
                description=hit["description"],
                chunk_index=hit["chunk_index"],
                max_chunk_size=hit["max_chunk_size"],
                overlap=hit["overlap"],
            )
            results.append(
                {
                    "chunk_id": hit["chunk_id"],
                    "distance": hit["distance"],
                    "chunk": chunk_text,
                    "movie_id": hit["movie_id"],
                    "title": hit["title"],
                    "description": hit["description"],
                }
            )

        return results

    def _reconstruct_chunk_text(
        self,
        title: str,
        description: str,
        chunk_index: int,
        max_chunk_size: int,
        overlap: int,
    ) -> str:
        """
        Rebuild the chunk text from (title, description, chunk_index, max_chunk_size, overlap)
        without storing the chunk in the DB.

        Convention:
          - chunk_index == 0: title
          - chunk_index >= 1: semantic_chunk(description, max_chunk_size, overlap)[chunk_index - 1]
        """
        if chunk_index == 0:
            return title

        desc_chunks = semantic_chunk(description, max_chunk_size, overlap)
        idx = chunk_index - 1
        if idx < 0 or idx >= len(desc_chunks):
            # fallback: whole description if something went wrong
            return description
        return "".join(desc_chunks[idx])

    # ------------------------ Verify helpers ------------------------ #
    def verify_model(self) -> None:
        model = self.model
        print(f"Model loaded: {model}")
        print(f"Max sequence length: {model.max_seq_length}")
        print(f"Embedding dim: {self.embedding_dim}")

    def verify_db(self) -> None:
        cur = self.conn.cursor()

        cur.execute("SELECT COUNT(*) FROM movies")
        (movie_count,) = cur.fetchone()

        cur.execute("SELECT COUNT(*) FROM chunks")
        (chunk_count,) = cur.fetchone()

        cur.execute("SELECT COUNT(*) FROM chunk_embeddings")
        (vec_count,) = cur.fetchone()

        cur.execute("SELECT vec_version()")
        (vec_version,) = cur.fetchone()

        print(f"Vector DB path: {self.db_path}")
        print(f"sqlite-vec version: {vec_version}")
        print(f"Movies count:            {movie_count}")
        print(f"Chunks table count:      {chunk_count}")
        print(f"Embeddings (vec0) count: {vec_count}")
        print(f"Embedding dim:           {self.embedding_dim}")
