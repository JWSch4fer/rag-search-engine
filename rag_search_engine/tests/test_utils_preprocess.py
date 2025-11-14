
import pytest
from rag_search_engine.utils import utils

@pytest.mark.parametrize("text,expected_contains", [
    ("The Matrix", {"matrix"}),                           # stopword removal + lowercase
    ("Hello, world!", {"hello", "world"}),               # punctuation stripped
    ("sci-fi", {"scifi"}),                               # hyphen collapsed
    ("running jumping", {"run", "jump"}),                # lemmatization
    ("watching windmills", {"watch", "windmill"}),       # lemmatization plural
    ("We go now.", {"go"}),                              # keep 'go' from allowlist
])
def test_preprocess_basics(text, expected_contains):
    tokens = utils.preprocess([text])[0]
    assert expected_contains.issubset(set(tokens))

def test_preprocess_accents_and_anime_mapping():
    # animated/anime variants should normalize together and accents folded
    tokens1 = utils.preprocess(["I love animated films"])[0]
    tokens2 = utils.preprocess(["classic animé film"])[0]
    assert "anime" in tokens1
    assert "anime" in tokens2 or "anime" in set(tokens2)  # allow either folded or canonical
    assert "film" in set(tokens1) | set(tokens2)

def test_preprocess_scifi_variants():
    t1 = utils.preprocess(["this is scifi / sci-fi"])[0]
    # accept sciencefiction or scifi depending on mapping; at least canonical present
    assert "sciencefiction" in t1 or "scifi" in t1
