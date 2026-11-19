"""Semantic recommendation components."""

from learner.recommendation.engine import SearchEngine, semantic_search
from learner.recommendation.embeddings import encode_query, cosine_scores

__all__ = ["SearchEngine", "semantic_search", "encode_query", "cosine_scores"]
