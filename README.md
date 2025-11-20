# rag-search-engine

Hybrid movie search engine with:

```
- **BM25 keyword search** (inverted index in SQLite)
- **Semantic vector search** using `sentence-transformers` + `sqlite-vec`
- **Hybrid ranking** (linear blend or reciprocal-rank fusion)
- Optional **Gemini** integration for:
  - Spell-checking / rewriting / expanding queries
  - LLM-based reranking of search results
```



rag-search ...
so you can build an index and run different kinds of searches from the terminal.

Features
### 1. Clean data model

**NOTE**: Input is a JSON file with a top-level "movies" array, e.g.:

json
```json
{
  "movies": [
    {
      "id": 1,
      "title": "The Matrix",
      "description": "A hacker discovers reality is a simulation."
    },
    {
      "id": 2,
      "title": "Toy Story",
      "description": "Toys come to life when humans are not around."
    }
  ]
}
```
### 2. Text normalization & tokenization

- Fixes weird encodings and HTML entities (fix_text)

- Normalizes Unicode & folds diacritics (e.g. café → cafe)

- Uses spaCy (en_core_web_sm) for: Tokenization and Lemmatization

- Normalizes tokens into a canonical movie vocabulary:

- This gives a consistent token space for both keyword and semantic indexing.

### 3. Keyword (BM25) search

- Builds a BM25 inverted index inside SQLite:

- Fast and works even without any ML/LLM dependencies.

### 4. Semantic search with sqlite-vec

- Uses SentenceTransformer to embed text.

- Uses sqlite-vec’s vec0 virtual table for vector search.

- Descriptions are sentence-chunked (semantic_chunk) with configurable:

- Results are per-movie, aggregating the best chunks:

### 5. Hybrid search & reranking

- Weighted search (weighted_search)

- Normalizes BM25 scores and semantic similarities into [0, 1].

    - 1.0 → pure keyword

    - 0.0 → pure semantic

    - 0.5 → 50/50

- Reciprocal Rank Fusion (rrf_search)

- Combines rank positions (not scores) from BM25 and semantic search:

- More robust when scores are on different scales or when one source is sparse.

- Both methods return enriched result dicts with per-source ranks and scores.

### 6. Gemini integration (optional)

- If you configure a GEMINI_API_KEY, you get:

- Query enhancement (Gemini.enhance):

- spell – fix obvious typos (“funy scifi movi” → “funny sci-fi movie”)

- rewrite – rephrase query to be clearer/cleaner

- expand – expand query with related terms

- Reranking (Gemini.rerank_document / Gemini.rerank_batch):

- individual – score each movie independently given the query (0–10 style prompt)

- batch – rank a whole list of candidates, returning IDs in relevance order

These use the official google-genai Python client (from google import genai).

**NOTE** ⚠️ LLM features are optional. The core search engine works without them.⚠️ 

## Installation

### 1. Install the package
From the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install .
pip install sentence-transformers sqlite-vec rapidfuzz python-dotenv google-genai
python -m spacy download en_core_web_sm
```
### 2. Configure Gemini (optional)
Either export an environment variable:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```
or create a .env file in the project root:
rag_search_engine.config calls load_dotenv() at import, so the CLI sees it automatically.
```bash
GEMINI_API_KEY="your-gemini-api-key"
```

### 3. Usage
Once installed, the CLI is available as:

All commands share the same SQLite DB path (by default):
```bash
rag-search build data/movies.json # Creates/updates the movies table.


rag-search key_search "animated toys" --limit 5
# Sample output:
# 
# text
# Copy code
#  1. 2.3456  Toy Story
#  2. 1.9876  Toy Story 2
#  ...

rag-search semantic_search "dream inside a dream heist" --limit 5
# Sample output (distance = lower is better):
# 
# text
# Copy code
#  1. 0.1234  Inception
#  2. 0.2345  Paprika
#  ...

rag-search hybrid_rrf "funy animted toyz" \
  --enhance spell \
  --limit 5
# RRF + query enhancement + LLM reranking:

```
## Running tests
```bash
pip install pytest
pytest
```
Notes:

Semantic-search tests require sentence-transformers and sqlite-vec; if those are missing, those tests will be skipped via pytest.importorskip.

Gemini tests require google-genai; they also skip if it’s not available.

## License

This project is licensed under the [MIT License](LICENSE).

