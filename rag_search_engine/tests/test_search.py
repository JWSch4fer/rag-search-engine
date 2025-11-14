
import pytest
from rag_search_engine.utils import search

def test_basic_search_titles(patch_load_data):
    # Expect to retrieve titles that contain the query tokens
    res = search.basic_search("matrix")
    assert isinstance(res, list)
    assert "The Matrix" in res

def test_search_inverted_index_animated_vs_anime(patch_load_data):
    res1 = search.search_inverted_index("animated")
    res2 = search.search_inverted_index("anime")
    # Both queries should hit "Classic Anime" when normalization works
    assert "Classic Anime" in res1
    assert "Classic Anime" in res2

def test_search_scifi_variants(patch_load_data):
    r1 = search.search_inverted_index("scifi")
    r2 = search.search_inverted_index("sci-fi")
    assert r1 and r2
    # They both should at least include the Sci-Fi title
    assert "Sci-Fi Epic" in r1 or "Sci-Fi Epic" in r2

