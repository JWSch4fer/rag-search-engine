import argparse
import json
import types

import pytest

from rag_search_engine.cli import cli as cli_mod


def test_make_parser_registers_subcommands():
    parser = cli_mod.make_parser()
    subparsers_actions = [
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    ]
    choices = set()
    for a in subparsers_actions:
        choices.update(a.choices.keys())

    for name in {
        "build",
        "key_search",
        "rrf-search",
        "semantic_search",
        "weighted-search",
    }:
        assert name in choices

    # sanity-check that func is wired for at least one subcommand
    ns = parser.parse_args(["build", "movies.json"])
    assert ns.func is cli_mod.handle_build
    assert ns.file_path == "movies.json"


class _DummyBase:
    def __init__(self, *args, **kwargs):
        self.args = (args, kwargs)
        self.verified = False
        self.closed = False

    def verify_db(self):
        self.verified = True

    def close(self):
        self.closed = True


def test_handle_build_uses_build_and_verify(monkeypatch, tmp_path, capsys):
    created = {}

    class DummyKS(_DummyBase):
        @classmethod
        def build_from_docs(cls, docs_path, db_path=None, force=False):
            inst = cls(docs_path, db_path, force)
            created["ks"] = inst
            return inst

    class DummySSC(_DummyBase):
        @classmethod
        def build_from_docs(cls, docs_path, db_path=None, force=False):
            inst = cls(docs_path, db_path, force)
            created["ssc"] = inst
            return inst

    monkeypatch.setattr(cli_mod, "KeywordSearch", DummyKS)
    monkeypatch.setattr(cli_mod, "SemanticSearch", DummySSC)

    args = argparse.Namespace(file_path=str(tmp_path / "movies.json"), force=True)
    cli_mod.handle_build(args)

    out = capsys.readouterr().out
    assert "Keyword index DB path" not in out  # from real verify_db, not used here
    assert created["ks"].verified
    assert created["ks"].closed
    assert created["ssc"].verified
    assert created["ssc"].closed


def test_handle_keyword_search_prints_ranked_results(monkeypatch, capsys):
    class DummyKS:
        def __init__(self, *_, **__):
            pass

        def search(self, query, k=10, k1=1.5, b=0.75):
            return [
                {"title": "MovieA", "score": 1.23},
                {"title": "MovieB", "score": 0.75},
            ]

        def close(self):
            pass

    monkeypatch.setattr(cli_mod, "KeywordSearch", DummyKS)

    args = argparse.Namespace(query="bear movie", limit=2)
    cli_mod.handle_keyword_search(args)

    out = capsys.readouterr().out.strip().splitlines()
    assert out
    assert out[0].strip().startswith("1. 1.2300  MovieA")
    assert out[1].strip().startswith("2. 0.7500  MovieB")


def test_handle_semantic_search_prints_results(monkeypatch, capsys):
    class DummySSC:
        def __init__(self, *_, **__):
            pass

        def query_top_k(self, query, k=10, knn_multiplier=10):
            return [
                {"title": "MovieA", "distance": 0.12},
                {"title": "MovieB", "distance": 0.34},
            ]

        def close(self):
            pass

    monkeypatch.setattr(cli_mod, "SemanticSearch", DummySSC)

    args = argparse.Namespace(query="animated toys", limit=2)
    cli_mod.handle_semantic_search(args)

    out = capsys.readouterr().out.strip().splitlines()
    assert out[0].strip().startswith("1. 0.1200  MovieA")
    assert out[1].strip().startswith("2. 0.3400  MovieB")


def test_handle_hybrid_weight_uses_hybridsearch(monkeypatch, capsys):
    class DummyHybrid:
        def __init__(self, *_, **__):
            pass

        def weighted_search(self, query, alpha, limit):
            return [
                {"title": "MovieA", "score": 0.9},
                {"title": "MovieB", "score": 0.5},
            ]

        def close(self):
            pass

    monkeypatch.setattr(cli_mod, "HybridSearch", DummyHybrid)

    args = argparse.Namespace(query="sci fi", alpha=0.7, limit=2)
    cli_mod.handle_hybrid_weight(args)

    out = capsys.readouterr().out.strip().splitlines()
    assert out[0].strip().startswith("0.9000  MovieA")
    assert out[1].strip().startswith("0.5000  MovieB")


def test_handle_hybrid_rrf_with_enhancement_and_base_output(monkeypatch, capsys):
    created = {}

    class DummyGemini:
        def __init__(self):
            created["gi"] = self
            self.calls = []

        def enhance(self, method, query):
            self.calls.append((method, query))
            return f"{query} [enhanced:{method}]"

    class DummyHybrid:
        def __init__(self, *_, **__):
            pass

        def rrf_search(self, query, k, limit, rerank_method=None):
            return [
                {
                    "id": 1,
                    "title": "MovieA",
                    "description": "DescA",
                    "score": 0.8,
                    "bm25_rank": 0,
                    "sem_rank": 1,
                }
            ]

        def close(self):
            pass

    monkeypatch.setattr(cli_mod, "Gemini", DummyGemini)
    monkeypatch.setattr(cli_mod, "HybridSearch", DummyHybrid)

    args = argparse.Namespace(
        query="space movie",
        enhance="spell",
        k=60,
        limit=1,
        rerank_method=None,
    )
    cli_mod.handle_hybrid_rrf(args)

    out = capsys.readouterr().out
    # enhanced query printed
    assert (
        "Enhanced query (spell): 'space movie' -> 'space movie [enhanced:spell]'" in out
    )
    # base RRF output branch
    assert "0.8000  MovieA" in out
    assert created["gi"].calls == [("spell", "space movie")]
