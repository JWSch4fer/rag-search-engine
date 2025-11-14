
import pytest
from rag_search_engine.utils import search

def _mk_docmap():
    return {
        0: {"title": "alpha", "description": "alpha beta"},
        1: {"title": "beta", "description": "beta gamma"},
        2: {"title": "gamma", "description": "alpha gamma gamma"},
    }

def _mk_doclen(dm):
    return {k: len((v["title"] + " " + v["description"]).split()) for k,v in dm.items()}

def test_calc_idf():
    docmap = _mk_docmap()
    idf_alpha = search.calc_idf("alpha", docmap)
    idf_zeta = search.calc_idf("zeta", docmap)
    assert idf_alpha > idf_zeta  # seen terms should have reasonable idf vs unseen

def test_calc_bm25_and_freq():
    docmap = _mk_docmap()
    doclen = _mk_doclen(docmap)
    s0 = search.calc_bm25("alpha", 0, docmap, doclen, k1=1.2, b=0.75)
    s1 = search.calc_bm25("alpha", 1, docmap, doclen, k1=1.2, b=0.75)
    assert s0 > s1  # doc 0 mentions alpha more than doc 1

    s0f = search.calc_bm25_freq("alpha", 0, docmap, doclen, k1=1.2, b=0.75)
    assert s0f >= 0.0
