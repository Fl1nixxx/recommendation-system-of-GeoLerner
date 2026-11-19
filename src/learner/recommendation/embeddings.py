from __future__ import annotations

from typing import Any

import numpy as np


def encode_query(query: str, model: Any) -> np.ndarray:
    query = str(query or "").strip()

    if not query:
        raise ValueError("semantic query cannot be empty")

    vector = model.encode([f"query: {query}"],normalize_embeddings=True,convert_to_numpy=True)
    vector = np.asarray(vector, dtype=np.float32)

    if vector.ndim != 2 or vector.shape[0] != 1:
        raise ValueError("E5 model must return one 2D query embedding")

    return vector[0]


def cosine_scores(embeddings: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    vector = np.asarray(query_vector, dtype=np.float32)

    if matrix.ndim != 2 or vector.ndim != 1:
        raise ValueError("embeddings must be 2D and query_vector must be 1D")

    if matrix.shape[1] != vector.shape[0]:
        raise ValueError("Embedding dimensions do not match")

    return matrix @ vector
