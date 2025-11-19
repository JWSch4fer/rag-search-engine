# rag_search_engine/utils/basesearch_db.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

import sqlite3

from rag_search_engine.utils.utils import load_data
from rag_search_engine.config import DEFAULT_DB_PATH


class BaseSearchDB:
    """
    Base class that manages a shared SQLite database and the movies table.

    Schema (base layer):
      - movies(id INTEGER PRIMARY KEY, title TEXT, description TEXT)

    If docs_path is provided, it will load and sync the movies table.
    If docs_path is None, it will just open the DB for read/use without touching movies.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        docs_path: Path | str | None = None,
        force: bool = False,
    ) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.force = force

        self.docs_path: Optional[Path] = Path(docs_path) if docs_path else None
        self.documents: Optional[List[Dict[str, Any]]] = None
        if self.docs_path is not None:
            self.documents = load_data(self.docs_path)

        # SQLite connection
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

        # Base schema always exists
        self._init_base_schema()

        # Only sync movies table if we have documents
        if self.documents:
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
        if self.documents is None:
            return  # nothing to rebuild from

        cur = self.conn.cursor()
        cur.execute("DELETE FROM movies")

        rows = [(int(d["id"]), d["title"], d["description"]) for d in self.documents]
        cur.executemany(
            "INSERT INTO movies (id, title, description) VALUES (?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def _ensure_movies_synced(self) -> None:
        """Ensure movies table has same doc count as documents (or rebuild if forced)."""
        if self.documents is None:
            return

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

    # ---------------------- convenience constructors ----------------------
    @classmethod
    def build_from_docs(
        cls,
        docs_path: str | Path,
        db_path: str | Path | None = None,
        force: bool = False,
    ) -> "BaseSearchDB":
        """
        Build/sync the movies table from docs_path and return a ready DB object.
        """
        return cls(db_path=db_path, docs_path=docs_path, force=force)

    @classmethod
    def open_existing(
        cls,
        db_path: str | Path | None = None,
    ) -> "BaseSearchDB":
        """
        Open an existing DB
        Does not touch the movies table.
        """
        return cls(db_path=db_path, docs_path=None)
