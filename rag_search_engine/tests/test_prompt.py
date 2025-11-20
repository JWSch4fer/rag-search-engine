from rag_search_engine.llm.prompt import gemini_method, gemini_reranking


def test_gemini_method_spell_mode():
    q = "helo wrld"
    prompt = gemini_method("spell", q)
    assert "Fix any spelling errors" in prompt
    assert q in prompt
    assert "Corrected:" in prompt


def test_gemini_method_rewrite_mode():
    q = "funny space movie"
    prompt = gemini_method("rewrite", q)
    assert "Rewrite this movie search query" in prompt
    assert "Rewritten query:" in prompt


def test_gemini_method_expand_mode():
    q = "robot"
    prompt = gemini_method("expand", q)
    assert "Expand this movie search query" in prompt
    assert q in prompt


def test_gemini_method_unknown_returns_empty():
    prompt = gemini_method("unknown_mode", "anything")
    assert prompt == ""


def test_gemini_reranking_individual():
    q = "sci-fi about dreams"
    doc = {"id": 1, "title": "Inception", "description": "Dream heist movie."}
    prompt = gemini_reranking(q, doc, "individual")
    assert "Rate how well this movie matches the search query." in prompt
    assert "Score:" in prompt
    assert 'Inception' in prompt


def test_gemini_reranking_batch():
    q = "animated toys"
    docs = [
        {"id": 1, "title": "Toy Story", "description": "Toys come to life."},
        {"id": 2, "title": "Frozen", "description": "Ice powers."},
    ]
    prompt = gemini_reranking(q, docs, "batch")
    assert "Rank these movies by relevance" in prompt
    assert "Return ONLY the IDs in order of relevance" in prompt
    assert "ID: 1" in prompt and "ID: 2" in prompt


def test_gemini_reranking_unknown_returns_empty():
    prompt = gemini_reranking("q", [{"id": 1}], "weird_mode")
    assert prompt == ""
