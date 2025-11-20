import pytest

from rag_search_engine.utils import hybrid_search as hs_mod


class DummyKeywordSearch:
    def __init__(self, *_, **__):
        pass

    def search(self, query, k=10, k1=1.5, b=0.75):
        # ids: 1, 2, 3 with descending scores
        return [
            {"id": 1, "title": "Doc1", "description": "d1", "score": 3.0},
            {"id": 3, "title": "Doc3", "description": "d3", "score": 2.0},
            {"id": 2, "title": "Doc2", "description": "d2", "score": 1.0},
        ]

    def close(self):
        pass


class DummySemanticSearchOnly2:
    def __init__(self, *_, **__):
        pass

    def query_top_k(self, query_text, k=10, knn_multiplier=10):
        # distances, lower is better
        return [
            {
                "chunk_id": 0,
                "distance": 0.2,
                "chunk": "d2 chunk",
                "movie_id": 2,
                "title": "Doc2",
                "description": "d2",
            },
            {
                "chunk_id": 1,
                "distance": 0.4,
                "chunk": "d4 chunk",
                "movie_id": 4,
                "title": "Doc4",
                "description": "d4",
            },
        ]

    def close(self):
        pass


class DummyEmptySemantic:
    def __init__(self, *_, **__):
        pass

    def query_top_k(self, query_text, k=10, knn_multiplier=10):
        return []

    def close(self):
        pass


class DummyEmptyKeyword:
    def __init__(self, *_, **__):
        pass

    def search(self, query, k=10, k1=1.5, b=0.75):
        return []

    def close(self):
        pass


def _make_hybrid(monkeypatch, kw_cls, sem_cls, tmp_path):
    monkeypatch.setattr(hs_mod, "KeywordSearch", kw_cls)
    monkeypatch.setattr(hs_mod, "SemanticSearch", sem_cls)
    return hs_mod.HybridSearch(docs_path=None, db_path=tmp_path / "hybrid.db")


def test_weighted_search_combines_scores(monkeypatch, tmp_path):
    hs = _make_hybrid(monkeypatch, DummyKeywordSearch, DummySemanticSearchOnly2, tmp_path)
    try:
        results = hs.weighted_search("query", alpha=0.7, limit=10)
        assert results  # non-empty
        # scores in [0, 1]
        for r in results:
            assert 0.0 <= r["bm25"] <= 1.0
            assert 0.0 <= r["semantic"] <= 1.0
            expected = 0.7 * r["bm25"] + 0.3 * r["semantic"]
            assert pytest.approx(expected) == r["score"]
    finally:
        hs.close()


def test_weighted_search_degrades_to_bm25_when_no_semantic(monkeypatch, tmp_path):
    hs = _make_hybrid(monkeypatch, DummyKeywordSearch, DummyEmptySemantic, tmp_path)
    try:
        results = hs.weighted_search("query", alpha=0.8, limit=10)
        ids_by_score = [r["id"] for r in results]
        # Should match pure BM25 ordering: 1 (3.0), 3 (2.0), 2 (1.0)
        assert ids_by_score == [1, 3, 2]
    finally:
        hs.close()


def test_rrf_search_degrades_to_bm25_when_only_keyword(monkeypatch, tmp_path):
    hs = _make_hybrid(monkeypatch, DummyKeywordSearch, DummyEmptySemantic, tmp_path)
    try:
        results = hs.rrf_search("query", limit=3)
        ids = [r["id"] for r in results]
        assert ids == [1, 3, 2]
    finally:
        hs.close()


def test_rrf_search_degrades_to_semantic_when_only_semantic(monkeypatch, tmp_path):
    hs = _make_hybrid(monkeypatch, DummyEmptyKeyword, DummySemanticSearchOnly2, tmp_path)
    try:
        results = hs.rrf_search("query", limit=3)
        # With only semantic hits, order should be by distance ascending:
        # movie_id 2 (0.2), then 4 (0.4)
        ids = [r["id"] for r in results]
        assert ids[:2] == [2, 4]
    finally:
        hs.close()

