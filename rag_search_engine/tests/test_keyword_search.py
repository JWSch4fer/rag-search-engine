import json
from pathlib import Path

from rag_search_engine.utils.keyword_search import KeywordSearch


def _tiny_movies(tmp_path) -> Path:
    movies = {
        "movies": [
            {
                "id": 1,
                "title": "The Matrix",
                "description": "A computer hacker learns about the true nature of reality.",
            },
            {
                "id": 2,
                "title": "Inception",
                "description": "A thief enters dreams to steal secrets.",
            },
            {
                "id": 3,
                "title": "Toy Story",
                "description": "Toys come to life when humans are not around.",
            },
        ]
    }
    path = tmp_path / "movies.json"
    path.write_text(json.dumps(movies, ensure_ascii=False), encoding="utf-8")
    return path


def test_keyword_search_returns_ranked_results(tmp_path):
    docs_path = _tiny_movies(tmp_path)
    db_path = tmp_path / "kw.db"

    ks = KeywordSearch.build_from_docs(docs_path=docs_path, db_path=db_path, force=True)
    try:
        results = ks.search("Matrix", k=5)
        assert isinstance(results, list)
        assert results

        top = results[0]
        assert top["title"] == "The Matrix"
        assert "score" in top
        # scores sorted descending
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
    finally:
        ks.close()


def test_keyword_search_empty_query_returns_empty_list(tmp_path):
    docs_path = _tiny_movies(tmp_path)
    db_path = tmp_path / "kw2.db"

    ks = KeywordSearch.build_from_docs(docs_path=docs_path, db_path=db_path, force=True)
    try:
        results = ks.search("", k=5)
        assert results == []
    finally:
        ks.close()

