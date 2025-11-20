import json
import types

import pytest

pytest.importorskip("google.genai")

from rag_search_engine.llm import gemini as gem_mod  # noqa: E402


class DummyResponse:
    def __init__(self, text: str):
        self.text = text
        self.usage_metadata = types.SimpleNamespace(
            prompt_token_count=1,
            candidates_token_count=2,
            total_token_count=3,
        )


class DummyModels:
    def __init__(self, response: DummyResponse):
        self._response = response

    def generate_content(self, **kwargs):
        return self._response


class DummyClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # will be replaced per test
        self.models = DummyModels(DummyResponse("[]"))


def test_enhance_short_circuits_on_empty_query(monkeypatch):
    # Patch genai.Client to avoid any real network calls
    monkeypatch.setattr(
        gem_mod,
        "genai",
        types.SimpleNamespace(Client=DummyClient),
    )

    g = gem_mod.Gemini(api_key="fake-key")

    out = g.enhance(method="spell", query="   ")
    assert out == "   "  # must return original query unchanged


def test_rerank_batch_parses_json_ids(monkeypatch):
    def make_client(api_key: str):
        response = DummyResponse(json.dumps([3, 1, "2"]))
        return types.SimpleNamespace(models=DummyModels(response))

    monkeypatch.setattr(
        gem_mod,
        "genai",
        types.SimpleNamespace(Client=make_client),
    )

    g = gem_mod.Gemini(api_key="fake-key")

    docs = [
        {"id": 1, "title": "A", "description": "a"},
        {"id": 2, "title": "B", "description": "b"},
        {"id": 3, "title": "C", "description": "c"},
    ]

    ranked = g.rerank_batch("query", docs, method="batch")
    # ints only, preserving order from JSON
    assert ranked == [3, 1, 2]
