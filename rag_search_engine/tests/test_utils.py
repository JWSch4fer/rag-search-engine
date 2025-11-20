import unicodedata

from rag_search_engine.utils.utils import (
    fix_text,
    fold_diacritics,
    normalize_token_semantic,
    preprocess,
    chunk,
    semantic_chunk,
    min_max_norm,
    hybrid_score,
    rrf_score,
)
from rag_search_engine.config import CANONICAL_VOCAB


def test_fix_text_decodes_unicode_escape_and_html_entities():
    # contains a literal unicode escape and an HTML entity
    s = r"caf\u00e9 &amp; crème brûlée"
    out = fix_text(s)

    # JSON-style unicode escape should be decoded
    assert "café" in out
    # HTML entity should be decoded
    assert "&amp;" not in out and " & " in out
    # no literal \u00e9 left
    assert r"\u00e9" not in out
    # NFC normalization
    assert unicodedata.is_normalized("NFC", out)


def test_fold_diacritics_removes_accents():
    s = "café déjà vu"
    out = fold_diacritics(s)
    assert out == "cafe deja vu"
    # all combining marks removed
    assert all(not unicodedata.combining(ch) for ch in out)


def test_normalize_token_semantic_direct_map():
    # explicit mapping via NORMALIZATION_MAP
    norm = normalize_token_semantic("sci-fi")
    assert norm == "sciencefiction"
    assert norm in CANONICAL_VOCAB


def test_normalize_token_semantic_fuzzy_match():
    # deliberately misspelled common genre; should fuzzy-match
    norm = normalize_token_semantic("ccomedy")
    assert norm in CANONICAL_VOCAB
    # very likely standard canonical token
    assert norm == "comedy"


def test_preprocess_single_string_removes_stopwords_and_normalizes():
    texts = "The quick brown fox jumps over the lazy dog."
    tokens_list = preprocess(texts)
    assert isinstance(tokens_list, list)
    assert len(tokens_list) == 1

    tokens = tokens_list[0]
    # stopwords removed
    assert "the" not in tokens
    # content words present and lemmatized
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    assert "jump" in tokens or "jumps" in tokens  # depending on lemmatizer
    assert "dog" in tokens


def test_preprocess_list_of_strings():
    texts = ["The Matrix", "Toy Stories"]
    tokens_list = preprocess(texts)
    assert len(tokens_list) == 2
    assert any("matrix" in t for t in tokens_list[0])
    # plural → lemma
    assert any("story" in t for t in tokens_list[1])


def test_chunk_with_overlap():
    words = "one two three four five six seven".split()
    chunks = chunk(words, chunk_size=3, overlap=1)

    # first chunk is the first 3 words
    assert chunks[0] == ["one", "two", "three"]
    # overlap: second chunk starts with last word of previous chunk
    assert chunks[1][0] == "three"
    # chunk size respected
    assert all(1 <= len(c) <= 3 for c in chunks)


def test_chunk_string_input_equivalent_to_list():
    text = "one two three four five"
    from_str = chunk(text, chunk_size=2, overlap=1)
    from_list = chunk(text.split(), chunk_size=2, overlap=1)
    assert from_str == from_list


def test_semantic_chunk_sentence_overlap():
    text = "Sentence one. Sentence two! Sentence three?"
    chunks = semantic_chunk(text, max_chunk_size=2, overlap=1)

    # should be in sentence units
    assert all(isinstance(c, list) for c in chunks)
    assert chunks[0] == ["Sentence one.", "Sentence two!"]
    assert chunks[1] == ["Sentence two!", "Sentence three?"]
    # overlap of 1 sentence between chunks
    assert chunks[0][1] == chunks[1][0]


def test_min_max_norm_scales_between_0_and_1():
    nums = [10.0, 20.0, 30.0]
    out = min_max_norm(nums)
    assert out[0] == 0.0
    assert out[-1] == 1.0
    assert all(0.0 <= x <= 1.0 for x in out)


def test_min_max_norm_all_equal_returns_all_ones():
    nums = [5.0, 5.0, 5.0]
    out = min_max_norm(nums)
    assert out == [1.0, 1.0, 1.0]


def test_hybrid_score_interpolation_limits():
    # pure BM25
    assert hybrid_score(1.0, 0.0, alpha=1.0) == 1.0
    # pure semantic
    assert hybrid_score(1.0, 0.0, alpha=0.0) == 0.0
    # 50/50 mix
    assert hybrid_score(1.0, 0.0, alpha=0.5) == 0.5


def test_rrf_score_monotonically_decreases_with_rank():
    assert rrf_score(0) > rrf_score(1) > rrf_score(10)

