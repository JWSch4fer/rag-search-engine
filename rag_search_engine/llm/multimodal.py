from __future__ import annotations

from typing import List, Dict, Optional

import json
from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    """
    Wrapper around a CLIP model from sentence-transformers to generate
    image embeddings and compare them with text embeddings for movies.

    Default model: 'clip-ViT-B-32'
    """

    def __init__(
        self,
        documents: Optional[List[Dict[str, object]]] = None,
        model_name: str = "clip-ViT-B-32",
    ) -> None:
        # Load the CLIP model
        self.model = SentenceTransformer(model_name)

        # Documents and derived text representation
        self.documents: List[Dict[str, object]] = documents or []
        self.texts: List[str] = []
        self.text_embeddings: Optional[np.ndarray] = None

        if self.documents:
            for doc in self.documents:
                title = str(doc.get("title", "")).strip()
                desc = str(doc.get("description", doc.get("document", ""))).strip()
                self.texts.append(f"{title}: {desc}")

            # Encode all movie texts with CLIP
            self.text_embeddings = self.model.encode(
                self.texts,
                convert_to_numpy=True,
                show_progress_bar=True,
            )

    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Generate an embedding for an image at the given path.

        - Loads the image using PIL.Image.open
        - Passes the image to the CLIP model's encode method
        - Returns a 1D numpy array (embedding vector)
        """
        image = Image.open(image_path).convert("RGB")

        embeddings = self.model.encode([image], convert_to_numpy=True)
        return embeddings[0]

    def search_with_image(
        self,
        image_path: str,
        top_k: int = 5,
    ) -> List[Dict[str, object]]:
        """
        Perform an image-to-text search:

        - Embed the image
        - Compute cosine similarity between image embedding and each
          movie text embedding
        - Return the top_k results as a list of dicts containing:
          id, title, description, similarity
        """
        if not self.documents or self.text_embeddings is None:
            raise ValueError(
                "MultimodalSearch was initialized without documents; "
                "cannot run search_with_image."
            )

        # Embed the query image
        image_emb = self.embed_image(image_path)

        # Normalize embeddings for cosine similarity
        image_vec = image_emb / (np.linalg.norm(image_emb) + 1e-12)

        text_vecs = self.text_embeddings
        text_norms = np.linalg.norm(text_vecs, axis=1, keepdims=True)
        text_normed = text_vecs / (text_norms + 1e-12)

        # Cosine similarity: dot(normalized_text, normalized_image)
        similarities = (text_normed @ image_vec).astype(float)

        # Get top_k indices sorted by similarity descending
        top_k = min(top_k, len(self.documents))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results: List[Dict[str, object]] = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append(
                {
                    "id": doc.get("id", idx),
                    "title": doc.get("title", ""),
                    "description": doc.get("description", doc.get("document", "")),
                    "similarity": float(similarities[idx]),
                }
            )

        return results


def verify_image_embedding(image_path: str) -> None:
    """
    Convenience function:
    - Create a MultimodalSearch instance (no documents needed)
    - Generate an embedding for the image
    - Print the shape in the required format:

        Embedding shape: <D> dimensions
    """
    ms = MultimodalSearch(documents=None)
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(
    image_path: str,
    limit: int = 5,
) -> List[Dict[str, object]]:
    """
    Top-level helper:

    - Load the movie dataset
    - Build a MultimodalSearch instance
    - Run search_with_image and return the results
    """
    root = Path(__file__).resolve().parents[1]
    movies_path = root / "data" / "movies.json"

    with movies_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Support either a list at top-level or a dict with a 'movies' key
    if isinstance(data, dict):
        documents = data.get("movies") or data.get("results") or []
    else:
        documents = data

    ms = MultimodalSearch(documents=documents)
    return ms.search_with_image(image_path, top_k=limit)
