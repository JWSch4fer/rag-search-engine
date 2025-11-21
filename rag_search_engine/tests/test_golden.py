import json
from pathlib import Path

import pytest

from rag_search_engine.utils.hybrid_search import HybridSearch

EVAL_K = 10


@pytest.fixture(scope="session")
def golden_cases():
    """
    Load test cases from data/golden_dataset.json.

    Expected structure:
    {
      "test_cases": [
        {"query": "...", "relevant_docs": ["Title1", "Title2", ...]},
        ...
      ]
    }
    """
    # tests/ -> project root -> data/golden_dataset.json
    root = Path(__file__).resolve().parents[2]
    golden_path = root / "data" / "golden_dataset.json"
    assert golden_path.exists(), f"Missing golden dataset at {golden_path}"

    with golden_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict) and "test_cases" in data
    cases = data["test_cases"]
    assert isinstance(cases, list)
    return cases


@pytest.fixture(scope="session")
def hybrid_search(tmp_path_factory):
    """
    Build a temporary HybridSearch index over data/movies.json
    for evaluation against the golden test cases.
    """
    root = Path(__file__).resolve().parents[2]
    docs_path = root / "data" / "movies.json"
    assert docs_path.exists(), f"Missing movies data at {docs_path}"

    db_dir = tmp_path_factory.mktemp("rrf_eval_db")
    db_path = db_dir / "movies.db"

    hs = HybridSearch(docs_path=docs_path, db_path=db_path, force=False)
    try:
        yield hs
    finally:
        hs.close()


def test_rrf_cross_encoder_on_golden_dataset(hybrid_search, golden_cases, capsys):
    """
    Evaluate the RRF + cross-encoder pipeline on the golden dataset.

    For each query:
      - run rrf_search with k=60 and limit=EVAL_K using the "cross_encoder" reranker
      - compute Precision@EVAL_K and Recall@EVAL_K
      - require Recall@EVAL_K > 0.0 (at least one relevant doc retrieved)
      - print metrics in a human-readable format (similar to the old CLI)
    """
    print(f"k={EVAL_K}\n")

    for case in golden_cases:
        query = str(case.get("query", "")).strip()
        assert query, "Golden dataset case has an empty query"

        relevant_titles = [str(t) for t in case.get("relevant_docs", [])]
        assert relevant_titles, f"No relevant_docs listed for query {query!r}"

        hits = hybrid_search.rrf_search(
            query,
            k=60,
            limit=EVAL_K,
            rerank_method="cross_encoder",
        )

        retrieved_titles = [
            str(doc.get("title", "")).strip()
            for doc in hits
            if doc.get("title")
        ]

        num_relevant = sum(1 for t in retrieved_titles if t in relevant_titles)
        denom_precision = len(retrieved_titles) or 1
        denom_recall = len(relevant_titles) or 1

        precision = num_relevant / denom_precision
        recall = num_relevant / denom_recall

        retrieved_str = ", ".join(retrieved_titles)
        relevant_str = ", ".join(relevant_titles)

        print(f"- Query: {query}")
        print(f"  - Precision@{EVAL_K}: {precision:.4f}")
        print(f"  - Recall@{EVAL_K}: {recall:.4f}")
        print(f"  - Retrieved: {retrieved_str}")
        print(f"  - Relevant: {relevant_str}")
        print()

        # Require at least one relevant doc in the top-K results
        assert recall > 0.0, (
            f"Recall@{EVAL_K} is 0 for query {query!r}. "
            f"Retrieved={retrieved_titles}, Relevant={relevant_titles}"
        )

    # Ensure prints are flushed; use `pytest -s` to see them
    _ = capsys.readouterr()

