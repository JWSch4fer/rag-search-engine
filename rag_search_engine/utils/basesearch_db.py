# rag_search_engine/utils/base_search_db.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import sqlite3

from rag_search_engine.utils.utils import ROOT, load_data


DEFAULT_DB_PATH = ROOT / "cache" / "movies.db"


class BaseSearchDB:
    """
    Base class that manages a shared SQLite database and the movies table.

    Schema (base layer):
      - movies(id INTEGER PRIMARY KEY, title TEXT, description TEXT)
    Subclasses (SemanticSearch, KeywordSearch) add their own tables/indexes.
    """

    def __init__(self, docs_path: Path | str, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.docs_path = Path(docs_path)

        # Load documents once; subclasses can reuse
        self.documents: List[Dict[str, Any]] = load_data(self.docs_path)

        # SQLite connection
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

        # Base schema
        self._init_base_schema()
        self._ensure_movies_synced()

    # ---------------------- base schema ---------------------- #
    def _init_base_schema(self) -> None:
        """Create the base movies table if it doesn't exist."""
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
        self.conn.commit()

    def _rebuild_movies_table(self) -> None:
        """Wipe and rebuild movies table from self.documents."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM movies")

        rows = [
            (int(d["id"]), d["title"], d["description"])
            for d in self.documents
        ]
        cur.executemany(
            "INSERT INTO movies (id, title, description) VALUES (?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def _ensure_movies_synced(self) -> None:
        """Ensure movies table has same doc count as documents."""
        n_docs = len(self.documents)
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM movies")
        (existing,) = cur.fetchone()

        if existing != n_docs:
            self._rebuild_movies_table()

    # ---------------------- helpers ---------------------- #
    def count_movies(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM movies")
        (count,) = cur.fetchone()
        return count

    def close(self) -> None:
        self.conn.close()

