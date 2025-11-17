# rag_search_engine/cli/lib/semantic_search.py
from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import sqlite3
import sqlite_vec

import numpy as np
import numpy.typing as npt

from sentence_transformers import SentenceTransformer

from rag_search_engine.utils.utils import load_data, semantic_chunk

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = CACHE_DIR / "movies_vec.db"
CHUNK_DB_PATH = CACHE_DIR / "movies_chunk_vec.db"


class SemanticSearch:
    """
    sqlite-vec–backed semantic search over movie docs.

    Schema:
      - movies(id INTEGER PRIMARY KEY, title TEXT, description TEXT)
      - movie_embeddings (vec0 virtual table):
            embedding float[dim] distance_metric=cosine
        rowid of movie_embeddings == movies.id
    """

    def __init__(self) -> None:
        # Local embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # SQLite + sqlite-vec
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._load_vec_extension()
        self._init_schema()

    # ------------------------ SQLite / schema ------------------------ #
    def _load_vec_extension(self) -> None:
        """Load sqlite-vec into this SQLite connection."""
        self.conn.enable_load_extension(True)
        sqlite_vec.load(
            self.conn
        )  # loads vec_version(), vec0, etc. :contentReference[oaicite:2]{index=2}
        self.conn.enable_load_extension(False)

    def _init_schema(self) -> None:
        """Create base + vec0 tables if they don't exist."""
        cur = self.conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS movies (
                id          INTEGER PRIMARY KEY,
                title       TEXT NOT NULL,
                description TEXT NOT NULL
            )
            """
        )

        # vec0 table storing embeddings; cosine distance for semantic similarity
        cur.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS movie_embeddings
            USING vec0(
              embedding float[{self.embedding_dim}] distance_metric=cosine
            )
            """
        )
        self.conn.commit()

    # ------------------------ Build / Sync ------------------------ #
    def build_or_sync(self, documents: List[Dict[str, Any]]) -> None:
        """
        Ensure the DB matches the provided documents.

        For now, if counts differ we wipe+rebuild; if counts match we assume in sync.
        """
        n_docs = len(documents)
        cur = self.conn.cursor()

        cur.execute("SELECT COUNT(*) FROM movies")
        row = cur.fetchone()
        existing = row[0] if row else 0

        if existing == n_docs:
            # Simple heuristic: assume DB already matches docs
            return

        # Rebuild from scratch
        cur.execute("DELETE FROM movies")
        cur.execute("DELETE FROM movie_embeddings")
        self.conn.commit()

        # Prepare text corpus: "title: description"
        texts: List[str] = [f"{d['title']}: {d['description']}" for d in documents]

        # Batch-encode all documents
        embeddings = self.model.encode(texts, show_progress_bar=True)
        emb_arr: npt.NDArray[np.float32] = np.asarray(embeddings, dtype=np.float32)

        # Insert into movies table
        movie_rows = [
            (i, d["title"], d["description"]) for i, d in enumerate(documents)
        ]
        cur.executemany(
            "INSERT INTO movies (id, title, description) VALUES (?, ?, ?)",
            movie_rows,
        )

        # Insert into vec0 table; rowid == movies.id
        for i, vec in enumerate(emb_arr):
            cur.execute(
                "INSERT INTO movie_embeddings(rowid, embedding) VALUES (?, ?)",
                (i, vec),
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
        self, query_text: str, k: int = 5
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Return top-k results as (orig_id, cosine_similarity, metadata).

        We use:
          - vec0 KNN: WHERE embedding MATCH :query AND k = :k
          - distance_metric=cosine, so sqlite-vec gives us cosine *distance*
            and we convert to similarity = 1 - distance. :contentReference[oaicite:3]{index=3}
        """
        query_vec = self.generate_embedding(query_text)[0]

        cur = self.conn.cursor()
        cur.execute(
            """
            WITH knn AS (
                SELECT rowid, distance
                FROM movie_embeddings
                WHERE embedding MATCH :query_vec
                  AND k = :k
            )
            SELECT
                m.id,
                m.title,
                m.description,
                knn.distance
            FROM knn
            JOIN movies AS m
              ON m.id = knn.rowid
            ORDER BY knn.distance ASC
            """,
            {
                "query_vec": query_vec,
                "k": int(k),
            },
        )

        rows = cur.fetchall()

        results: List[Tuple[int, float, Dict[str, Any]]] = []
        for doc_id, title, desc, dist in rows:
            # cosine distance -> similarity
            sim = 1.0 - float(dist)
            meta = {"title": title, "description": desc}
            results.append((int(doc_id), sim, meta))

        return results

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

        cur.execute("SELECT COUNT(*) FROM movie_embeddings")
        (vec_count,) = cur.fetchone()

        cur.execute("SELECT vec_version()")
        (vec_version,) = cur.fetchone()

        print(f"Vector DB path: {DB_PATH}")
        print(f"sqlite-vec version: {vec_version}")
        print(f"Movies table count:      {movie_count}")
        print(f"Embeddings (vec0) count: {vec_count}")
        print(f"Embedding dim:           {self.embedding_dim}")


# ------------------------ Top-level helpers for CLI ------------------------ #
def verify_model() -> None:
    s = SemanticSearch()
    s.verify_model()


def verify_embeddings(file_path: str | Path) -> None:
    """
    For parity with your previous 'verify_embeddings' command:
    - Load movies.json (via load_data)
    - Build/sync into the sqlite-vec DB
    - Print corpus size and embedding dimensionality
    """
    s = SemanticSearch()
    data = load_data(file_path)
    documents = data  # assuming load_data already returns the list of movies
    s.build_or_sync(documents)

    cur = s.conn.cursor()
    cur.execute("SELECT COUNT(*) FROM movies")
    (n_docs,) = cur.fetchone()

    print(f"Number of docs:   {n_docs}")
    print(f"Embeddings shape: {n_docs} vectors in " f"{s.embedding_dim} dimensions")


def vdb_query(query: str, k: int = 5) -> None:
    s = SemanticSearch()
    results = s.query_top_k(query, k=k)

    for rank, (orig_id, sim, meta) in enumerate(results, start=1):
        title = meta.get("title", f"doc {orig_id}")
        print(f"{rank:2d}. {sim:.4f}  {title}")


class SemanticSearchChunked:
    """
    sqlite-vec–backed semantic search over chunked movie docs.

    Schema:
      - chunk_movies(id INTEGER PRIMARY KEY, title TEXT, description TEXT)
      - chunks (id INTEGER PRIMARY KEY, movie_id FK(chunk_movies.id), chunk TEXT)
      - chunk_embeddings (vec0 virtual table):
            embedding float[dim] distance_metric=cosine
        rowid of movie_embeddings == movies.id
    """

    def __init__(self) -> None:
        # Local embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # SQLite + sqlite-vec for semantic chunked data
        self.conn_chunk = sqlite3.connect(CHUNK_DB_PATH)
        self.conn_chunk.execute("PRAGMA journal_mode=WAL")
        self.conn_chunk.execute("PRAGMA synchronous=NORMAL")
        self._chunk_load_vec_extension()
        self._chunk_init_schema()

    # ------------------------ SQLite / schema ------------------------ #
    def _chunk_load_vec_extension(self) -> None:
        """Load sqlite-vec into this SQLite connection."""
        self.conn_chunk.enable_load_extension(True)
        sqlite_vec.load(
            self.conn_chunk
        )  # loads vec_version(), vec0, etc. :contentReference[oaicite:2]{index=2}
        self.conn_chunk.enable_load_extension(False)

    def _chunk_init_schema(self) -> None:
        """Create chunk tables + vec0 virtual table if they don't exist."""
        cur = self.conn_chunk.cursor()

        # One row per movie
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_movies (
                id          INTEGER PRIMARY KEY,
                title       TEXT NOT NULL,
                description TEXT NOT NULL       -- full movie description
            )
            """
        )

        # Many chunks per movie
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id           INTEGER PRIMARY KEY,
                movie_id     INTEGER NOT NULL,  -- FK to chunk_movies.id
                chunk        TEXT NOT NULL,
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

        self.conn_chunk.commit()

    # ------------------------ Build / Sync ------------------------ #
    def build_or_sync_chunk(
        self, documents: List[Dict[str, Any]], max_chunk_size: int, overlap: int
    ) -> None:
        """
        Ensure the DB matches the provided documents.

        For now, if counts differ we wipe+rebuild; if counts match we assume in sync.
        """
        n_docs = len(documents)
        cur = self.conn_chunk.cursor()

        cur.execute("SELECT COUNT(*) FROM chunk_movies")
        row = cur.fetchone()
        existing = row[0] if row else 0

        if existing == n_docs:
            # Simple heuristic: assume DB already matches docs
            return

        # Rebuild from scratch
        cur.execute("DELETE FROM chunk_movies")
        cur.execute("DELETE FROM chunks")
        cur.execute("DELETE FROM chunk_embeddings")
        self.conn_chunk.commit()

        # Prepare text: "title: description"
        chunk_idx = 0
        for idx, d in enumerate(documents):
            title = d["title"]
            description = d["description"]

            cur = self.conn_chunk.cursor()

            cur.execute(
                "INSERT INTO chunk_movies (id, title, description) VALUES (?,?,?)",
                (idx, title, description),
            )

            # title is its own chunk
            chunks = [(chunk_idx, idx, title)]
            chunk_idx += 1
            # description chunks
            for c_text in semantic_chunk(description, max_chunk_size, overlap):
                chunks += [(chunk_idx, idx, "".join(c_text))]
                chunk_idx += 1

            # NOTE: chunks.id == chunk_embeddings.rowid
            cur.executemany(
                "INSERT INTO chunks (id, movie_id, chunk) VALUES (?,?,?)", chunks
            )
        cur.execute("SELECT id, chunk FROM chunks")
        rows = cur.fetchall()  # [(id, chunk_text),...]
        embed_text = [r[1] for r in rows]
        # embed the chunk
        embed_ids = [r[0] for r in rows]
        embed_text = self.model.encode(embed_text, show_progress_bar=True)
        cur.executemany(
            "INSERT INTO chunk_embeddings(rowid, embedding) VALUES (?,?)",
            zip(embed_ids, embed_text),
        )
        self.conn_chunk.commit()

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
        Return top-k results as (orig_id, cosine_similarity, metadata).

        We use:
          - vec0 KNN: WHERE embedding MATCH :query AND k = :k
          - distance_metric=cosine, so sqlite-vec gives us cosine *distance*
            and we convert to similarity = 1 - distance. :contentReference[oaicite:3]{index=3}
        """
        query_vec = self.model.encode([query_text]).astype("float32")[0]

        cur = self.conn_chunk.cursor()
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

        results = [
            {
                "chunk_id": rowid,
                "distance": distance,
                "chunk": chunk,
                "movie_id": movie_id,
                "title": title,
                "description": description,
            }
            for (rowid, distance, chunk, movie_id, title, description) in rows
        ]
        return results

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

        cur.execute("SELECT COUNT(*) FROM movie_embeddings")
        (vec_count,) = cur.fetchone()

        cur.execute("SELECT vec_version()")
        (vec_version,) = cur.fetchone()

        print(f"Vector DB path: {DB_PATH}")
        print(f"sqlite-vec version: {vec_version}")
        print(f"Movies table count:      {movie_count}")
        print(f"Embeddings (vec0) count: {vec_count}")
        print(f"Embedding dim:           {self.embedding_dim}")


# ______________________________________________________________________________
def verify_chunk_embeddings(
    file_path: str | Path, max_chunk_size: int, overlap: int
) -> None:
    """
    For parity with your previous 'verify_embeddings' command:
    - Load movies.json (via load_data)
    - Build/sync into the sqlite-vec DB
    - Print corpus size and embedding dimensionality
    """
    ssc = SemanticSearchChunked()
    documents = load_data(file_path)
    ssc.build_or_sync_chunk(documents, max_chunk_size, overlap)

    cur = ssc.conn_chunk.cursor()
    cur.execute("SELECT COUNT(*) FROM chunk_movies")
    (n_docs,) = cur.fetchone()
    cur.execute("SELECT COUNT(*) FROM chunks")
    (n_chunks,) = cur.fetchone()

    print(f"Number of docs:   {n_docs}")
    print(f"Number of chunks: {n_chunks}")
    print(f"Embeddings shape: {n_chunks} vectors in " f"{ssc.embedding_dim} dimensions")


def search_chunked(text: str, limit: int) -> None:
    """
    search the chunked database for the text
    """
    ssc = SemanticSearchChunked()
    results = ssc.query_top_k(text, limit)
    for d in results:
        print(d["movie_id"], d["title"])
