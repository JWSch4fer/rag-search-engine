# rag_search_engine/cli/lib/semantic_search.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy.typing as npt
import numpy as np

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer

from rag_search_engine.utils.utils import load_data

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CHROMA_PATH = CACHE_DIR / "chroma"
COLLECTION_NAME = "movies"
# Ensure cosine distance; Chroma returns "distances" for this space
COLLECTION_METADATA = {"hnsw:space": "cosine"}


class SemanticSearch:
    """
    Vector-DB backed semantic search over movie docs.
    """

    def __init__(self) -> None:
        # Keep the model around for verification and optional manual embedding
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Let Chroma do embedding by itself via an embedding function
        self._embed_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Persistent client + collection
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata=COLLECTION_METADATA,
        )

    # ------------------------ Build / Sync ------------------------ #
    def build_or_sync(self, documents: List[Dict[str, Any]]) -> None:
        """
        Ensure the collection matches the provided documents: if counts mismatch
        or missing ids are detected, (re)upsert everything.
        """
        # Our external IDs (string) are the dataset index; adjust to your own IDs if needed.
        ids = [str(i) for i in range(len(documents))]
        docs = [f"{d['title']}: {d['description']}" for d in documents]
        metas = [
            {"title": d["title"], "description": d["description"], "orig_id": i}
            for i, d in enumerate(documents)
        ]

        # Heuristic: if collection count != documents count → rebuild
        col_count = self.collection.count()
        if col_count != len(documents):
            # Simplest approach: recreate the collection
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self._embed_fn,
                metadata=COLLECTION_METADATA,
            )

            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
            return

        # Counts match — check if every id exists (lightweight integrity check)
        # Chroma doesn’t have a fast “contains” per id; simplest is to upsert all (idempotent)
        # self.collection.upsert(ids=ids, documents=docs, metadatas=metas)

    # ------------------------ Query ------------------------ #
    def generate_embedding(self, text: str | List[str]) -> npt.NDArray[np.float64]:
        # deal with single string or list of strings
        match text:
            case str():
                text = [text]

        for t in text:  # no string can be empty
            if len(t.strip()) == 0:
                raise ValueError("cannot embed empty text")

        return self.model.encode(text)

    def query_top_k(
        self, query_text: str, k: int = 5
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Return top-k results as (orig_id, cosine_similarity, metadata).
        Chroma returns distances; for cosine space, similarity = 1 - distance.
        """
        res = self.collection.query(
            query_texts=[query_text],
            n_results=k,
            include=["distances", "metadatas", "documents"],
        )
        # Unpack results from Chroma's list-of-lists format
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        out: List[Tuple[int, float, Dict[str, Any]]] = []
        for id_str, dist, meta in zip(ids, dists, metas):
            # cosine similarity = 1 - cosine distance
            sim = 1.0 - float(dist)
            orig_id = (
                int(meta.get("orig_id")) if meta and "orig_id" in meta else int(id_str)
            )
            out.append((orig_id, sim, meta or {}))

        # Already ranked by Chroma
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    # ------------------------ Verify helpers ------------------------ #
    def verify_model(self) -> None:
        model = self.model
        print(f"Model loaded: {model}")
        print(f"Max sequence length: {model.max_seq_length}")
        print(f"Embedding dim: {model.get_sentence_embedding_dimension()}")

    def verify_db(self) -> None:
        print(f"Vector DB path: {CHROMA_PATH}")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Count: {self.collection.count()}")


# ------------------------ Top-level helpers for CLI ------------------------ #
def verify_model() -> None:
    s = SemanticSearch()
    s.verify_model()


def verify_embeddings(file_path: str | Path) -> None:
    """
    For parity with your previous 'verify_embeddings' command:
    - Load movies.json
    - Build/sync into the vector DB
    - Print corpus size and embedding dimensionality
    """
    s = SemanticSearch()
    data = load_data(file_path)
    documents = data
    s.build_or_sync(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {len(documents)} vectors in "
        f"{s.model.get_sentence_embedding_dimension()} dimensions"
    )


def vdb_query(query: str, k: int = 5) -> None:
    s = SemanticSearch()

    results = s.query_top_k(query, k=k)
    for rank, (orig_id, sim, meta) in enumerate(results, start=1):
        title = meta.get("title", f"doc {orig_id}")
        print(f"{rank:2d}. {sim:.4f}  {title}")

