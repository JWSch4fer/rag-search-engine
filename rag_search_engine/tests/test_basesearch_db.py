import json
from pathlib import Path

from rag_search_engine.utils.basesearch_db import BaseSearchDB


def _write_tiny_movies(tmp_path) -> Path:
    movies = {
        "movies": [
            {
                "id": 1,
                "title": "The Matrix",
                "description": "A hacker discovers that reality is a simulation.",
            },
            {
                "id": 2,
                "title": "Toy Story",
                "description": "Toys come to life when humans are not around.",
            },
        ]
    }
    path = tmp_path / "movies.json"
    path.write_text(json.dumps(movies, ensure_ascii=False), encoding="utf-8")
    return path


def test_build_from_docs_creates_movies_table(tmp_path):
    docs_path = _write_tiny_movies(tmp_path)
    db_path = tmp_path / "movies.db"

    db = BaseSearchDB.build_from_docs(docs_path=docs_path, db_path=db_path, force=True)
    try:
        assert db.count_movies() == 2
    finally:
        db.close()


def test_open_existing_does_not_touch_movies(tmp_path):
    docs_path = _write_tiny_movies(tmp_path)
    db_path = tmp_path / "movies.db"

    db1 = BaseSearchDB.build_from_docs(docs_path=docs_path, db_path=db_path, force=True)
    db1.close()

    # open_existing should reuse the same DB without resyncing docs
    db2 = BaseSearchDB.open_existing(db_path=db_path)
    try:
        assert db2.count_movies() == 2
    finally:
        db2.close()

