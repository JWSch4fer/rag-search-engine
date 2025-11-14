
import pytest

@pytest.fixture
def tiny_corpus():
    # minimal 4-doc corpus
    return {
        "movies": [
            {"title": "The Matrix", "description": "A hacker discovers the truth about reality."},
            {"title": "Classic Anime", "description": "An animated tale of heroes."},
            {"title": "Sci-Fi Epic", "description": "A science fiction adventure in space."},
            {"title": "We Go Now", "description": "We go now into the night."}
        ]
    }

@pytest.fixture
def patch_load_data(monkeypatch, tiny_corpus):
    # Patch search.load_data and utils.load_data to use the tiny corpus
    import rag_search_engine.utils.utils as utils_mod
    import rag_search_engine.utils.search as search_mod

    def _fake_load_data():
        return tiny_corpus

    monkeypatch.setattr(utils_mod, "load_data", _fake_load_data, raising=True)
    monkeypatch.setattr(search_mod, "load_data", _fake_load_data, raising=True)
    return _fake_load_data
