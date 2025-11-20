import json
from pathlib import Path

import numpy as np
import pytest

# These tests assume you have the optional deps installed
pytest.importorskip("sqlite_vec")
pytest.importorskip("sentence_transformers")

from rag_search_engine.utils import semantic_search as ss_mod  # noqa: E402


def _tiny_movies(tmp_path: Path) -> Path:
    movies = {
        "movies": [
            {
                "id": 1,
                "title": "The Matrix",
                "description": "A hacker learns reality is a simulation.",
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


class DummyModel:
    def __init__(self, *_, **__):
        self._dim = 4

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts, show_progress_bar=True):
        # deterministic small vectors
        return np.ones((len(texts), self._dim), dtype="float32")


def test_generate_embedding_uses_sentence_transformer(monkeypatch, tmp_path):
    # Patch only the SentenceTransformer so we avoid heavy model downloads.
    # We let the real sqlite_vec extension load so vec0 is available.
    monkeypatch.setattr(ss_mod, "SentenceTransformer", DummyModel)

    docs_path = _tiny_movies(tmp_path)
    db_path = tmp_path / "sem.db"

    ss = ss_mod.SemanticSearch(
        docs_path=docs_path,
        db_path=db_path,
        max_chunk_size=1,
        overlap=0,
        force=True,
    )
    try:
        vec = ss.generate_embedding("hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (1, 4)
        assert np.all(vec == 1.0)
    finally:
        ss.close()
