from __future__ import annotations

import numpy as np
import pandas as pd


def rank_indices(indices: np.ndarray,scores: np.ndarray,metadata: pd.DataFrame,*,reference_mode: str | None = None,) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)

    if reference_mode != "author":
        return indices[np.argsort(scores[indices])[::-1]]

    candidates = metadata.iloc[indices].copy()

    count_column = next((column for column in ("ratings_count", "rating_count") if column in candidates),None,)
    rating_column = next((column for column in ("ratings_average", "rating", "source_rating") if column in candidates),None,)
    
    candidates["_rating_count"] = (pd.to_numeric(candidates[count_column], errors="coerce").fillna(0)if count_column else 0)
    candidates["_rating"] = (pd.to_numeric(candidates[rating_column], errors="coerce").fillna(0)if rating_column else 0)

    return candidates.sort_values(["_rating_count", "_rating"], ascending=False, kind="stable").index.to_numpy(dtype=np.int64)
